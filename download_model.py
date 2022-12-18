import gdown

url = "https://drive.google.com/file/d/1E8Mw6JcJpAsLj96z4A1op4n8sSQ7oeb-/view?usp=share_link"
output = "ckpt4_2C.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/1qvOiRNX9lO-Y-a_MvZm2nrc_-S6a32gD/view?usp=share_link"
output = "p1_checkpoints_blender.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)