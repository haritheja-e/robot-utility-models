from api_keys import OPENAI_KEY
from openai import OpenAI
import base64
import os
import time

MODEL = "gpt-4o"

TASK_CONFIG = {
    "door_opening": {
        "prompt": """
        As the timesteps progress, does the robotic arm open the door AND is the robot arm grasping the handle in the LAST timestep? 
        Please respond with only 'Yes' or 'No'. 
        """, 
        "camera": "head",
    },
    "drawer_opening": {
        "prompt": """
        As the timesteps progress, does the robotic arm grasp the drawer handle and open it AND is the drawer open in the last timestep? 
        Please respond with only 'Yes' or 'No'. 
        """, 
        "camera": "head",
    },
    "reorientation": {
        "prompt": """
        As the timesteps progress, does the robotic arm/gripper reorient the object upright AND is the object upright in the LAST frame? 
        Please respond with only 'Yes' or 'No'. 
        """, 
        "camera": "wrist",
    },
    "tissue_pick_up": {
        "prompt": """
        As the timesteps progress, does the robotic arm/gripper grasp the tissue AND is the gripper grasping the tissue in the LAST timestep? 
        Please respond with only 'Yes' or 'No'. 
        """, 
        "camera": "wrist",
    },
    "bag_pick_up": {
        "prompt": """
        As the timesteps progress, does the robotic arm/gripper grasp the bag AND is the gripper grasping the bag in the LAST timestep? 
        Please respond with only 'Yes' or 'No'. 
        """, 
        "camera": "wrist",
    },
}

class OpenAIClient:
    def __init__(self, task, img_save_dir):
        self.client = OpenAI(api_key=OPENAI_KEY)
        self.task = task
        self.img_save_dir = img_save_dir

        self.prompt = TASK_CONFIG[task]["prompt"]
        self.camera = TASK_CONFIG[task]["camera"]

    def get_image_list(self):
        latest_img_folder = sorted(os.listdir(self.img_save_dir))[-1]

        if self.camera == "wrist":
            img_path = os.path.join(self.img_save_dir, latest_img_folder)
        elif self.camera == "head":
            img_path = os.path.join(self.img_save_dir, latest_img_folder, "head_cam")
        else:
            raise ValueError(f"Invalid camera type: {self.camera}")
        
        img_files = [file for file in os.listdir(img_path) if file.endswith(".jpg")]
        img_files = sorted(img_files, key=lambda x: int(x.split(".")[0]))
        img_files = [os.path.join(img_path, file) for file in img_files]

        return img_files
    
    def get_encoded_images(self, step_n):
        image_file_paths = self.get_image_list()
        if not len(image_file_paths) == step_n + 1:
            time.sleep(1)
            image_file_paths = self.get_image_list()
            print("Waiting for all images to be saved...")
        
        # take every other image, including the last image, from head_cam_img_files list
        image_file_paths = image_file_paths[::-2][::-1]
        encoded_images = []
        for path in image_file_paths:
            with open(path, 'rb') as image_file:
                encoded_images.append(base64.b64encode(image_file.read()).decode('utf-8'))

        return encoded_images

    def get_image_prompts(self, encoded_images):
        image_prompts = []
        for idx, encoded_image in enumerate(encoded_images):
            image_prompts.append(
                {"type": "text", 
                 "text": f"The following is an image taken at timestep {idx}"
                }
            )
            image_prompts.append(
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{encoded_image}"}
                }
            )
        return image_prompts
    
    def get_response(self, step_n):
        encoded_images = self.get_encoded_images(step_n)
        image_prompts = self.get_image_prompts(encoded_images)

        completion = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": self.prompt}]
                },
                {
                    "role": "user",
                    "content": image_prompts
                }
            ],
            temperature=0.0,
        )

        response = completion.choices[0].message.content

        return response