# Mezo - AI-Powered Mesophase Analysis Application.

Mezo designed for managing and analyzing samples of pitch and their associated images. It provides a
user-friendly interface for viewing a library of samples and editing images.

![Mezo analysis](https://github.com/donatorex/AI_mezo/blob/main/assets/mezo_analysis.png?raw=true)

## Disclaimer

The Mezo program was developed as a personal project and is part of my portfolio. Please note that
support for this project is not guaranteed. If you have any questions about commercial use or other
collaboration, please contact me via the details provided in my GitHub profile.

## Before you start

1. Download the SAM model checkpoint (select ViT-H SAM model) from the github repository at
[this link](https://github.com/facebookresearch/segment-anything/blob/main/README.md#model-checkpoints.);
2. Place the downloaded model checkpoint (file .pth) in the root folder;
3. Make sure the file name matches the `sam_checkpoint` constant in the
[mezo/editor.py](https://github.com/donatorex/AI_mezo/blob/main/mezo/editor.py) file - otherwise,
change the constant.

## Running

To run the Mezo program, please follow these steps:

1. **Navigate to the Program Directory:**

   First, change your current directory to the folder containing the Mezo program:

   ```bash
   cd path/to/program
   ```

   Replace `path/to/program` with the actual path to the directory where the program is located.

2. **Create a Virtual Environment:**

   Create a virtual environment to manage the dependencies for this project. You can do this by
   running the following command in your terminal:

   ```bash
   python -m venv venv
   ```

   This will create a new directory named `venv` containing the virtual environment.

3. **Activate the Virtual Environment:**

   - On Windows, activate the virtual environment with:

     ```bash
     .\venv\Scripts\activate
     ```

   - On macOS and Linux, use:

     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies:**

   With the virtual environment activated, install the required dependencies by running:

   ```bash
   pip install -r requirements.txt
   ```

   This will install all necessary packages listed in the `requirements.txt` file.



5. **Run the Program:**

   Finally, execute the program using the following command:

   ```bash
   flet run main.py
   ```

   This will start the program and you should see it running as expected.
