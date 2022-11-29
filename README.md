# DLCV-Fall-2022-HW4

Please click [this link](https://docs.google.com/presentation/d/171DwrrzYUenLnyev_NyZg0c19lgqk4q42iA_ptLZjDk/edit?usp=sharing) to view the slides of HW4

# Usage

To start working on this assignment, you should clone this repository into your local machine by using the following command.
    
    git clone https://github.com/DLCV-Fall-2022/hw4-benlin1211.git

### Create environment
You can run the following command to install all the packages listed in the requirements.txt:

    conda create --name dlcv-hw4 python=3.8
    conda activate dlcv-hw4
    pip3 install -r requirements.txt


### List all environments

    conda info --envs
    
### Check all package environment

    conda list -n dlcv-hw4

### Close an environment

    conda deactivate

### Remove an environment

    conda env remove -n dlcv-hw4
    



# Data
please download `hw4_data` from the link below
https://drive.google.com/file/d/1Tc0f28syYVE185Z6388DWUOVHGSANCmX/view?usp=share_link

# Submission Rules
### Deadline
2022/12/12 (Mon.) 23:59

### Packages
This homework should be done using python3.8. For a list of packages you are allowed to import in this assignment, please refer to the `requirements.txt` for more details.

You can run the following command to install all the packages listed in the requirements.txt:
``` Shell
pip install -r requirements.txt --no-cache-dir && pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu102.html --no-cache-dir
```

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.

# Q&A
If you have any problems related to HW4, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under hw4 FAQ section in FB group.(But TAs won't answer your question on FB.)

# Reminder:
inference: 

1. device = "cuda"
2. torch.load(model_name, map_location='cuda')

Example:

    if torch.cuda.is_available():
        if torch.cuda.device_count()==2:
            device = torch.device("cuda:1")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using", device)
    ...
    netG.load_state_dict(torch.load(ckpt_path, map_location='cuda'))
