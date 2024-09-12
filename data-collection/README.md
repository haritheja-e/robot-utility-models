## Usage

For extracting a single environment:

Ensure you've created the `home_robot` mamba environment from the first 3 steps of the [Imitation in Homes documentation](https://educated-diascia-662.notion.site/Setting-Up-Running-Zero-Shot-Models-on-Hello-Robot-Stretch-66658ab1a6454f219e0fb1db1baa9d6f?pvs=97#55e4606db0e045ada791177caa599692). 

1.  Compress video taken from the Record3D app:

    ![Export Data](https://github.com/user-attachments/assets/2c22358e-d0ad-4e18-8058-556156235e8a)
2. Get the files on your machine.
   1. **Option 1: Using Google drive:**
      1. \[Only once] Generate Google Service Account API key to download from private folders on Google Drive. There are some instructions on how to do so in this Stackoverflow link [https://stackoverflow.com/a/72076913](https://stackoverflow.com/a/72076913)
      2. \[Only once] Rename the .json file to `client_secret.json` and put it in the same directory as  `gdrive_downloader.py`
      3. Upload `.zip` file into its own folder on Google Drive, and copy folder\_id from URL to put it in the `GDRIVE_FOLDER_ID` in the `./do-all.sh` file.
   2. **Option 2: Manually**:
      *   Comment out the `GDRIVE_FOLDER_ID` line from `./do-all.sh` and create the following hierarchy locally

          ```bash
          dataset/
          |--- task1/
          |------ home1/
          |--------- env1/
          |------------ {data_file}.zip
          |--------- env2/
          |------------ {data_file}.zip
          |--------- env.../
          |------------ {data_file}.zip
          |------ home2/
          |------ home.../
          |--- task2/
          |--- task.../
          ```
      * The .zip files should contain .r3d files exported from the Record3D app in the previous step.
3. Modify required variables in `do-all.sh`.
   1. `TASK_NAME` task name.
   2. `HOME` name or ID of the home.
   3. `ROOT_FOLDER` folder where the data is stored after downloading.
   4. `EXPORT_FOLDER` folder where the dataset is stored after processing. Should be different from `ROOT_FOLDER`.
   5. `ENV_NO` current environment number in the same home and task set.
   6. `GRIPPER_MODEL_PATH` path to the gripper model. It should be in this folder as `gripper_model_new.pth`.
4.  Run

    ```bash
    ./do-all.sh
    ```