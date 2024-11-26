import time
from typing import Dict

import cv2
import numpy as np
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

def get_home_param(h=0.5, y=0.02, x=0.0, yaw=0.0, pitch=0.0, roll=0.0, gripper=1.0):
    """
    Returns a list of home parameters
    """
    return [h, y, x, yaw, pitch, roll, gripper]


schedule = None
LOCALHOST = "127.0.0.1"
ANYCAST = "0.0.0.0"
TIME_HORIZON = 15

class Controller:
    def __init__(self, cfg=None):
        global schedule

        self.cfg = cfg
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
        self.flag_socket = create_request_socket(
            network_cfg.get("remote", LOCALHOST), port=network_cfg["flag_port"]
        )

        self.publisher = publisher
        self.subscriber = subscriber

        self.head_cam_subscriber = None
        self.use_head_cam = cfg.get("use_head_cam")
        if self.use_head_cam:
            self.head_cam_subscriber = ZMQCameraSubscriber(
                network_cfg.get("remote", LOCALHOST),
                network_cfg["head_camera_port"],
                network_cfg.get("mode", "RGB" if not self.use_depth else "RGBD"),
            )

        if not self.use_depth:
            self.async_saver = AsyncImageActionSaver(cfg["image_save_dir"], cfg)
        else:
            self.async_saver = AsyncImageDepthActionSaver(cfg["image_save_dir"], cfg)

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

        if cfg.get("use_gpt") is True:
            from robot.openai_client import OpenAIClient
            self.openai_client = OpenAIClient(cfg["task"], cfg["image_save_dir"])

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

    def _query_gpt(self, step_n):
        if self.head_cam_subscriber is None: 
            return RuntimeError("Set \'use_head_cam\' config to True") 
        print("Querying GPT")
        
        responses = self.openai_client.get_response(step_n)
        print(responses)
        return responses

    def _run_policy(self, run_for=1, use_gpt=False):
        while run_for > 0:
            cv2_img, timestamp = self.subscriber.recv_rgb_image()
            print("time to receive image:", time.time() - timestamp)
            self.image_action_buffer_manager.add_image(cv2_img)
            if self.use_head_cam:
                cv2_head_img, timestamp_head = self.head_cam_subscriber.recv_rgb_image()
                self.image_action_buffer_manager.save_head_img(cv2_head_img)

            if use_gpt:
                if time.time() - self.start_time > TIME_HORIZON:
                    response = self._query_gpt(self.step_n)
                    if "no" in response.lower():
                        print("Task failed.")
                        return "retry"
                    elif "yes" in response.lower():
                        print("Task completed.")
                        return "reset"

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

                action_tensor, logs = self.model.step(input_tensor_sequence, step_no=self.step_n)
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
            print("Gripper:", gripper)

            if not self.abs_gripper:
                self.gripper = self.gripper + gripper
                gripper = self.gripper

            final_action = np.append(action_robot, gripper)
            # publish action to robot_action topic
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

    def _run(self, run_for=1):
        if not self.use_depth:
            self._run_policy(run_for=run_for)
        else:
            self._run_policy_depth(run_for=run_for)

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
            self.publisher.pub_keypoints(get_home_param(h=self.h, x=None), "params")
        elif instruction.lower() == "s":
            sched_no = input("Enter schedule number:")
            base, h = schedule(int(sched_no))
            print(h, base)
            self.publisher.pub_keypoints(get_home_param(h=h, x=base), "params")
            self.schedul_no = int(sched_no)
        elif instruction.lower() == "t":
            self.publisher.pub_keypoints(get_home_param(h=self.h, x="home"), "params")
        elif instruction.lower() == "n":
            self.schedul_no += 1
            base, h = schedule(self.schedul_no)
            print(h, base)
            self.publisher.pub_keypoints(get_home_param(h=h, x=base), "params")
        elif len(instruction) == 0:
            self.run_for = 1
            self._run(self.run_for)
        elif instruction.lower() == "q":
            self.async_saver.finish()
            exit()
        else:
            # raise warning
            print("Invalid instruction")
            instruction = input("Enter instruction:")
            self._process_instruction(instruction)
    
    def self_eval(self, run_no):
        print("Running Schedule Location", run_no+1, "on Trial", self.trial_no+1)
        base, h = schedule(int(run_no+1))
        print(h, base)
        self.publisher.pub_keypoints(get_home_param(h=h, x=base), "params")
        self.flag_socket.recv()
        self.flag_socket.send(b"")
        self.publisher.pub_keypoints([1], "home")
        self.flag_socket.recv()
        self.flag_socket.send(b"")
        time.sleep(2)
        self.start_time = time.time()
        while True:
            result = self._run_policy(run_for=1, use_gpt=True)

            if result == "retry":
                print("Retrying")
                time.sleep(2)
                self.reset_experiment()
                self.publisher.pub_keypoints(get_home_param(h=self.h, x="home"), "params")
                self.flag_socket.recv()
                self.flag_socket.send(b"")
                self.publisher.pub_keypoints([1], "home")
                self.flag_socket.recv()
                self.flag_socket.send(b"")
                time.sleep(6)
                run_no = np.random.randint(0, 10)
                self.self_eval(run_no=run_no)
                break
            elif result == "reset":
                print("Resetting")
                time.sleep(2)
                self.reset_experiment()
                self.publisher.pub_keypoints(get_home_param(h=self.h, x="home"), "params")
                self.flag_socket.recv()
                self.flag_socket.send(b"")
                self.publisher.pub_keypoints([1], "home")
                self.flag_socket.recv()
                time.sleep(8)
                break

            self.flag_socket.recv()
            self.flag_socket.send(b"")

    def run(self):
        # send flag before sending the first instruction
        self.flag_socket.send(b"")
        time.sleep(0.5)
        self.publisher.pub_keypoints(get_home_param(h=self.h), "params")
        self.flag_socket.recv()

        while True:
            self.flag_socket.send(b"")
            
            # wait for instruction
            instruction = input("Enter instruction:")
            
            if instruction.lower() == "q":
                instruction = self._process_instruction(instruction)
                break
            elif instruction.isdigit():
                run_for = int(instruction)
                for _ in range(run_for - 1):
                    self._run()
                    self.flag_socket.recv()
                    self.flag_socket.send(b"")
                self._run()
                self.flag_socket.recv()
            elif instruction.lower() == "e":
                num_runs = 10
                for i in range(num_runs):
                    self.trial_no = i
                    self.self_eval(run_no=i)
                    if i != num_runs - 1:
                        self.flag_socket.send(b"")
            else:
                # process and send instruction to robot
                instruction = self._process_instruction(instruction)
                # continue loop only once instruction has been executed on robot
                self.flag_socket.recv()
