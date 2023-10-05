import os
import sys

def test_floods():
    os.chdir("Instagan")
    os.system("python test.py --dataroot datasets/street2water --model instagan --name s2w2 --loadSizeH 320 --loadSizeW 320 --fineSizeH 320 --fineSizeW 320")

def test_heat_waves():
    os.system("git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git")
    os.system("cp -r pix2pix/* pytorch-CycleGAN-and-pix2pix/")
    os.chdir("pytorch-CycleGAN-and-pix2pix")
    os.system("python test.py --dataroot /path/to/your/data --name test --model cycle_gan --netG unet_256")

def test_storms():
    os.chdir("style transfer/style transfer")
    os.system("python style_transfer.py -c ./07.jpg -s ./cloudd.jpg -save out07 -steps 400 -style_weight 180000 -sharpness_weight 1.0")
    # Add similar lines for other images as needed

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <project_name>")
        sys.exit(1)
    
    project_name = sys.argv[1].lower()
    
    if project_name == 'floods':
        test_floods()
    elif project_name == 'heat_waves':
        test_heat_waves()
    elif project_name == 'storms':
        test_storms()
    else:
        print(f"Unknown project name: {project_name}")
