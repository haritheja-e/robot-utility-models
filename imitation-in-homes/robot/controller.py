from enum import Enum
import logging
import time
from typing import Dict

import cv2
import numpy as np
import gradio as gr
import torch
from scipy.spatial.transform import Rotation as R
import wandb

from utils.action_transforms import invert_permutation_transform
from utils.zmq_utils import (
    ZMQKeypointPublisher,
    ZMQCameraSubscriber,
    create_request_socket,
)
from .utils import (
    AsyncImageActionSaver,
    AsyncImageDepthActionSaver,
    ImageActionBufferManager,
    ImageDepthActionBufferManager,
    schedule_init,
)

logger = logging.getLogger(__name__)
STRETCH_GRIPPER_MAX = 150


def get_home_param(
    h=0.5,
    y=0.02,
    x=0.0,
    yaw=0.0,
    pitch=0.0,
    roll=0.0,
    gripper=1.0,
    closing_threshold=0.5,
    reopening_threshold=0.5,
    stretch_gripper_max=None,
    stretch_gripper_min=None,
    stretch_gripper_tight=None,
    sticky_gripper=None,
    # Below the first value, it will close, above the second value it will open
    gripper_threshold_post_grasp_list=None,
):
    """
    Returns a list of home parameters
    """
    return [
        h,
        y,
        x,
        yaw,
        pitch,
        roll,
        gripper,
        stretch_gripper_max,
        stretch_gripper_min,
        stretch_gripper_tight,
        sticky_gripper,
        closing_threshold,
        reopening_threshold,
        gripper_threshold_post_grasp_list,
    ]


schedule = None
LOCALHOST = "127.0.0.1"
ANYCAST = "0.0.0.0"
task_to_params_dict = {
    "door_opening": {
        "max_gripper": 0.33,
        "gripper_threshold": 0.2,
        "opening_threshold": 1,
    },
    "drawer_opening": {
        "max_gripper": 0.5,
        "gripper_threshold": 0.2,
        "opening_threshold": 1,
    },
    "reorientation": {
        "max_gripper": 1.0,
        "gripper_threshold": 0.4,
        "opening_threshold": 0.6,
    },
    "bag_pick_up": {
        "max_gripper": 1.0,
        "gripper_threshold": 0.3,
        "opening_threshold": 1,
    },
    "tissue_pick_up": {
        "max_gripper": 1.0,
        "gripper_threshold": 0.2,
        "opening_threshold": 1,
    },
}


class StartingPositions(Enum):
    HOME = 0
    POSITION_1 = 1
    POSITION_2 = 2
    POSITION_3 = 3
    POSITION_4 = 4
    POSITION_5 = 5
    POSITION_6 = 6
    POSITION_7 = 7
    POSITION_8 = 8
    POSITION_9 = 9
    POSITION_10 = 10


class Controller:
    def __init__(self, cfg=None):
        global schedule

        self.cfg = cfg
        self.task = cfg["task"]
        self.use_depth = cfg["use_depth"]

        network_cfg: Dict = cfg["network"]
        publisher = ZMQKeypointPublisher(
            network_cfg.get("host", ANYCAST), network_cfg["action_port"]
        )
        subscriber = ZMQCameraSubscriber(
            network_cfg.get("remote", LOCALHOST),
            network_cfg["camera_port"],
            network_cfg.get("mode", "RGB" if not self.use_depth else "RGBD"),
        )
        self._ui_subscriber = ZMQCameraSubscriber(
            network_cfg.get("remote", LOCALHOST),
            network_cfg["camera_port"],
            network_cfg.get("mode", "RGB" if not self.use_depth else "RGBD"),
        )
        self.flag_socket = create_request_socket(
            network_cfg.get("remote", LOCALHOST), port=network_cfg["flag_port"]
        )

        self.publisher = publisher
        self.subscriber = subscriber

        if not self.use_depth:
            self.async_saver = AsyncImageActionSaver(cfg["image_save_dir"])
        else:
            self.async_saver = AsyncImageDepthActionSaver(cfg["image_save_dir"])

        self.image_action_buffer_manager = self.create_buffer_manager()

        self.device = cfg["device"]
        schedule = schedule_init(
            self,
            max_h=cfg["robot_params"]["max_h"],
            max_base=cfg["robot_params"]["max_base"],
        )

        self.run_n = -1
        self.step_n = 0
        self.schedul_no = -1
        self.h = cfg["robot_params"]["h"]

        self.abs_gripper = cfg["robot_params"]["abs_gripper"]
        self.gripper = 1.0
        self.rot_unit = cfg["robot_params"]["rot_unit"]

        if cfg.get("goal_conditional") is True:
            import sentence_transformers

            model = sentence_transformers.SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            self._goal_conditional = True
            self._embedding = model.encode([cfg["goal_description"]])
            self._embedding = torch.tensor(self._embedding).to(self.device)
            del model
        else:
            self._goal_conditional = False

        self._set_values_by_task(cfg["task"])
        self.demo = self._init_demo()

    def setup_model(self, model):
        self.model = model
        self.model.to(self.device)
        self.model.eval()

    def create_buffer_manager(self):
        if self.use_depth:
            return ImageDepthActionBufferManager(
                self.cfg["image_buffer_size"],
                self.async_saver,
                self.cfg["dataset"]["train"]["config"].get("depth_cfg"),
            )
        else:
            return ImageActionBufferManager(
                self.cfg["image_buffer_size"], self.async_saver
            )

    def action_tensor_to_matrix(self, action_tensor):
        affine = np.eye(4)
        if self.rot_unit == "euler":
            r = R.from_euler("xyz", action_tensor[3:6], degrees=False)
        elif self.rot_unit == "axis":
            r = R.from_rotvec(action_tensor[3:6])
        else:
            raise NotImplementedError
        affine[:3, :3] = r.as_matrix()
        affine[:3, -1] = action_tensor[:3]

        return affine

    def matrix_to_action_tensor(self, matrix):
        r = R.from_matrix(matrix[:3, :3])
        action_tensor = np.concatenate(
            (matrix[:3, -1], r.as_euler("xyz", degrees=False))
        )
        return action_tensor

    def cam_to_robot_frame(self, matrix):
        return invert_permutation_transform(matrix)

    def _update_log_keys(self, logs):
        new_logs = {}
        for k in logs.keys():
            new_logs[k + "_" + str(self.run_n)] = logs[k]

        return new_logs

    def _run_policy(self, run_for=1):
        while run_for > 0:
            cv2_img, timestamp = self.subscriber.recv_rgb_image()
            logger.info(f"time to receive image: {time.time() - timestamp}")
            self.image_action_buffer_manager.add_image(cv2_img)

            with torch.no_grad():
                input_tensor_sequence = (
                    self.image_action_buffer_manager.get_input_tensor_sequence()
                )

                if not self._goal_conditional:
                    input_tensor_sequence = (
                        input_tensor_sequence[0].to(self.device).unsqueeze(0),
                        input_tensor_sequence[1].to(self.device).unsqueeze(0),
                    )
                else:
                    input_tensor_sequence = (
                        input_tensor_sequence[0].to(self.device).unsqueeze(0),
                        self._embedding,
                        input_tensor_sequence[1].to(self.device).unsqueeze(0),
                    )

                action_tensor, logs = self.model.step(
                    input_tensor_sequence, step_no=self.step_n
                )
                if "indices" in logs:
                    indices = logs["indices"].squeeze()
                    for nbhr, idx in enumerate(indices):
                        img = self.model.train_dataset[idx]
                        img = (
                            (img[0][0]).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        )
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        self.async_saver.save_image(img, nbhr=nbhr)
                action_tensor = action_tensor.squeeze(0).cpu()
                self.image_action_buffer_manager.add_action(action_tensor)
                action_tensor = action_tensor.squeeze().numpy()

            action_matrix = self.action_tensor_to_matrix(action_tensor)
            action_robot_matrix = self.cam_to_robot_frame(action_matrix)
            action_robot = self.matrix_to_action_tensor(action_robot_matrix)

            gripper = action_tensor[-1]
            logger.info(f"Gripper: {gripper} {self.abs_gripper} {self.gripper}")

            if not self.abs_gripper:
                self.gripper = self.gripper + gripper
                gripper = self.gripper
            else:
                # Update member variable to be deplayed in the UI
                self.gripper = gripper

            final_action = np.append(action_robot, gripper)
            # publish action to robot_action topic
            logger.info("Publishing robot_action")
            self.publisher.pub_keypoints(final_action, "robot_action")

            wandb.log(self._update_log_keys(logs), step=self.step_n)
            run_for -= 1
            self.step_n += 1

    def _run_policy_depth(self, run_for=1):
        while run_for > 0:
            cv2_img, np_depth, timestamp = self.subscriber.recv_image_and_depth()
            self.image_action_buffer_manager.add_image(cv2_img)
            self.image_action_buffer_manager.add_depth(np_depth)

            with torch.no_grad():
                input_tensor_sequence = (
                    self.image_action_buffer_manager.get_input_tensor_sequence()
                )

                input_tensor_sequence = (
                    input_tensor_sequence[0].to(self.device).unsqueeze(0),
                    input_tensor_sequence[1].to(self.device).unsqueeze(0),
                    input_tensor_sequence[2].to(self.device).unsqueeze(0),
                )

                action_tensor, logs = self.model.step(input_tensor_sequence)
                action_tensor = action_tensor.squeeze(0).cpu()
                self.image_action_buffer_manager.add_action(action_tensor)
                action_tensor = action_tensor.squeeze().numpy()

            action_matrix = self.action_tensor_to_matrix(action_tensor)
            action_robot_matrix = self.cam_to_robot_frame(action_matrix)
            action_robot = self.matrix_to_action_tensor(action_robot_matrix)

            gripper = action_tensor[-1]

            if not self.abs_gripper:
                self.gripper = self.gripper + gripper
                gripper = self.gripper
            final_action = np.append(action_robot, gripper)
            self.publisher.pub_keypoints(final_action, "robot_action")

            wandb.log(self._update_log_keys(logs), step=self.step_n)
            run_for -= 1
            self.step_n += 1

    def _init_demo(self):
        # Convert the above demo to blocks.
        with gr.Blocks(theme="soft", analytics_enabled=False) as demo:
            gr.Markdown(
                """
            # Robot Evaluation Admin Tool
            1. Try to match the robot POV to the samples we provided. Generally, you will have to update the height and the initial gripper value. 
            To update robot runtime params, click "Update params", then "Home" to see the values reflected.
            2. Click "Take a policy step" when you're ready for the robot to execute.
            3. Repeat 2 as necessarily until the task is complete.
            4. Done!
            """
            )
            with gr.Row():
                with gr.Column(scale=1):
                    image = gr.Image(label="Robot POV")
                    gripper_val = gr.Number(
                        label="Gripper value",
                        show_label=True,
                        interactive=False,
                        value=self.get_gripper_val,
                        every=gr.Timer(0.03),
                    )
                    start_position_dropdown = gr.Dropdown(
                        [s.name for s in StartingPositions],
                        label="Start positions",
                        info="Choose a start position relative to the original position. Then 'Update Params' and 'Home' to get to that position.",
                    )

                with gr.Column(scale=4):
                    task_picker = gr.Radio(
                        label="Task ID",
                        info="This is only for choosing runtime params! To change the model you have to kill and restart the script.",
                        choices=task_to_params_dict.keys(),
                    )
                    height = gr.Slider(0, 1, value=0.5, step=0.01, label="Height")
                    gripper_close_threshold = gr.Slider(
                        0,
                        1,
                        value=self._gripper_threshold,
                        step=0.01,
                        label="Gripper closing threshold",
                        info="""
                        Once below the theshold value, gripper will close. 
                        If your robot is closing the gripper too early, make this lower. 
                        If your robot is closing the gripper too late, make this higher. 
                        You can look at the gripper value box on the left to get an idea of 
                        what the ideal closing threshold should be.""",
                    )
                    gripper_open_threshold = gr.Slider(
                        0,
                        1,
                        value=self._opening_threshold,
                        step=0.01,
                        label="Gripper opening threshold",
                        info="""
                        Once above the theshold value, gripper will open after closing.
                        Mostly important for tasks where the gripper closes and then opens 
                        again, like reorientation. 
                        If your robot is opening the gripper too early, make this lower. 
                        If your robot is opening the gripper too late, make this higher. 
                        You can look at the gripper value box on the left to get an idea of 
                        what the ideal closing threshold should be.""",
                    )
                    gripper = gr.Slider(
                        0,
                        1,
                        value=self._max_gripper,
                        step=0.01,
                        label="Gripper max width",
                        info="0.33 for door opening, 0.5 for drawer opening, 1 for everything else.",
                    )
                    gripper_tight = gr.Slider(
                        -0.5,
                        0.2,
                        value=-0.25,
                        step=0.01,
                        label="Gripper tightness",
                        info="This is how tight the gripper becomes when it closes. The lower the value, the tighter the gripper.",
                    )
                    task_picker.change(
                        fn=self._set_values_by_task,
                        inputs=[task_picker],
                        outputs=[
                            gripper,
                            gripper_close_threshold,
                            gripper_open_threshold,
                        ],
                    )

                    with gr.Row():
                        update_params_button = gr.Button(
                            "Update params", variant="secondary"
                        )
                        update_params_button.click(
                            fn=self.update_robot_params,
                            inputs=[
                                height,
                                gripper,
                                gripper_close_threshold,
                                gripper_open_threshold,
                                gripper_tight,
                                start_position_dropdown,
                            ],
                            outputs=None,
                        )
                        home_button = gr.Button("Home the robot")
                        home_button.click(fn=self._run_home)

                    step_button = gr.Button("Take a policy step", variant="primary")
                    step_button.click(fn=self._run)

            # On a timer, show the image.
            timer = gr.Timer(0.1)
            timer.tick(fn=self.get_image_vanilla, outputs=[image])

            # On a timer, get the gripper value
            # timer.tick(fn=self.get_gripper_val, outputs=[gripper_val])

        return demo

    def _set_values_by_task(self, task_name: str):
        self._opening_threshold = task_to_params_dict[task_name]["opening_threshold"]
        self._max_gripper = task_to_params_dict[task_name]["max_gripper"]
        self._gripper_threshold = task_to_params_dict[task_name]["gripper_threshold"]
        return self._max_gripper, self._gripper_threshold, self._opening_threshold

    def update_robot_params(
        self,
        height,
        gripper,
        gripper_close_threshold,
        gripper_open_threshold,
        gripper_tight,
        start_position_dropdown,
    ):
        logger.info(
            f"Publishing params: height={height}, gripper={gripper}, gripper_close_threshold={gripper_close_threshold}, gripper_open_threshold={gripper_open_threshold}, gripper_tight={gripper_tight}, starting_position={start_position_dropdown}"
        )

        # Map starting position back to the height and base position, then publish that location to move
        x: int = 0
        self.h = height
        if start_position_dropdown is not None:
            sched_no: int = StartingPositions[start_position_dropdown].value
            if sched_no != 0:
                base, h = schedule(sched_no)
                logger.info(f"Starting position {sched_no}: h={h}, base={base}")
                height = h
                x = base
                self.schedul_no = sched_no

        self.publisher.pub_keypoints(
            # Below the first value, it will close, above the second value it will open
            get_home_param(
                h=height,
                x=x,
                stretch_gripper_max=STRETCH_GRIPPER_MAX * gripper,
                closing_threshold=gripper_close_threshold,
                reopening_threshold=gripper_open_threshold,
                stretch_gripper_tight=STRETCH_GRIPPER_MAX * gripper_tight,
            ),
            "params",
        )
        self.flag_socket.recv()
        self.flag_socket.send(b"")

    def get_image_vanilla(self):
        while True:
            time.sleep(0.1)
            cv2_img, timestamp = self._ui_subscriber.recv_rgb_image()
            yield cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    def get_gripper_val(self):
        return self.gripper

    def _run_home(self):
        logger.info("Publishing home")
        self.publisher.pub_keypoints([1], "home")
        self.reset_experiment()
        self.flag_socket.recv()
        self.flag_socket.send(b"")

    def _run(self, run_for=1):
        logger.info(f"Run robot for {run_for} step(s)")
        if not self.use_depth:
            self._run_policy(run_for=run_for)
        else:
            self._run_policy_depth(run_for=run_for)
        # Reseting flag for command to be re-sent
        self.flag_socket.recv()
        self.flag_socket.send(b"")

    def reset_experiment(self):
        self.async_saver.finish()
        self.run_n += 1
        self.step_n = 0
        self.gripper = 1.0
        self.model.reset()
        self.image_action_buffer_manager = self.create_buffer_manager()

    def _process_instruction(self, instruction):
        global schedule
        if instruction.lower() == "h":
            self.publisher.pub_keypoints([1], "home")
            self.reset_experiment()

        elif instruction.lower() == "r":
            h = input("Enter height:")
            self.h = float(h)
            self.publisher.pub_keypoints(get_home_param(h=self.h), "params")

        elif instruction.lower() == "tl":
            threshold_low = input("Enter closing threshold: ")
            self.tl = float(threshold_low)
            self.publisher.pub_keypoints(
                get_home_param(h=self.h, closing_threshold=self.tl), "params"
            )

        elif instruction.lower() == "th":
            threshold_reopen = input("Enter reopening threshold: ")
            self.th = float(threshold_reopen)
            self.publisher.pub_keypoints(
                get_home_param(h=self.h, reopening_threshold=self.th), "params"
            )

        elif instruction.lower() == "mgw":
            max_gripper_width = input(
                "Enter max gripper width (155 for wide, 50 for low): "
            )
            self.mgw = float(max_gripper_width)
            self.publisher.pub_keypoints(
                get_home_param(h=self.h, stretch_gripper_max=self.mgw), "params"
            )

        elif instruction.lower() == "gt":
            gripper_tight_value = input(
                "Enter gripper value when closed tight (default -35): "
            )
            self.gt = float(gripper_tight_value)
            self.publisher.pub_keypoints(
                get_home_param(h=self.h, stretch_gripper_tight=self.gt), "params"
            )

        elif instruction.lower() == "sg":
            sticky_gripper_value = input(
                "Enter whether gripper should be sticky (i.e. close only once, default true): "
            )
            self.sg = bool(sticky_gripper_value)
            self.publisher.pub_keypoints(
                get_home_param(h=self.h, sticky_gripper=self.sg), "params"
            )

        elif instruction.lower() == "s":
            sched_no = input("Enter schedule number:")
            base, h = schedule(int(sched_no))
            logger.info(f"h={h}, base={base}")
            self.publisher.pub_keypoints(get_home_param(h=h, x=base), "params")
            self.schedul_no = int(sched_no)

        elif instruction.lower() == "n":
            self.schedul_no += 1
            base, h = schedule(self.schedul_no)
            logger.info(f"h={h}, base={base}")
            self.publisher.pub_keypoints(get_home_param(h=h, x=base), "params")
        elif len(instruction) == 0:
            self.run_for = 1
            self._run(self.run_for)
        elif instruction.isdigit():
            self.run_for = int(instruction)
            self._run(self.run_for)
        elif instruction.lower() == "q":
            self.async_saver.finish()
            exit()
        else:
            # raise warning
            logger.error("Invalid instruction")
            instruction = input("Enter instruction:")
            self._process_instruction(instruction)

    def run_continous(self):
        for i in range(20):
            logger.info(i)
            start_time = time.time()

            self.flag_socket.send(b"")

            self.run_for = 1
            self._run(self.run_for)

            self.flag_socket.recv()

            elapsed_time = time.time() - start_time
            sleep_time = max(0, 0.1 - elapsed_time)
            time.sleep(sleep_time)

        instruction = input("Enter instruction:")
        return instruction

    def run(self):
        # send flag before sending the first instruction
        self.flag_socket.send(b"")
        time.sleep(0.5)
        self.publisher.pub_keypoints(get_home_param(h=self.h), "params")
        self.flag_socket.recv()

        self.demo.launch(prevent_thread_lock=True, server_name="0.0.0.0", quiet=True)
        print("Launched the Gradio UI at http://ROBOT_IP:7860/")
        while True:
            self.flag_socket.send(b"")

            # wait for instruction
            instruction = input("Enter instruction: ")
            start_time = time.time()

            if instruction.lower() == "q":
                instruction = self._process_instruction(instruction)
                break
            elif instruction.lower() == "rc":
                self._run()
                self.flag_socket.recv()
                instruction = ""
                while len(instruction) == 0:
                    instruction = self.run_continous()
                continue

            # process and send instruction to robot
            instruction = self._process_instruction(instruction)

            # continue loop only once instruction has been executed on robot
            self.flag_socket.recv()

            elapsed_time = time.time() - start_time
            sleep_time = max(0, 0.1 - elapsed_time)
            time.sleep(sleep_time)
            # Calculate elapsed time and sleep to maintain 10 Hz frequency
