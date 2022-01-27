# mp4_converter_gui.py
import cv2
import glob
import os
import shutil
import PySimpleGUI as sg
from PIL import Image, ImageDraw
import numpy as np

file_types = [("MP4 (*.mp4)", "*.mp4"), ("All files (*.*)", "*.*")]

def convert_mp4_to_jpgs(path):
    video_capture = cv2.VideoCapture(path)
    still_reading, image = video_capture.read()
    frame_count = 0
    if os.path.exists("output"):
        # remove previous GIF frame files
        shutil.rmtree("output")
    try:
        os.mkdir("output")
    except IOError:
        sg.popup("Error occurred creating output folder")
        return
    
    while still_reading:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        npImage = np.array(image)
        h, w = image.size

        alpha = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(alpha)
        draw.pieslice([0, 0, h, w], 0, 360, fill = 255)

        npAlpha = np.array(alpha)
        npImage = np.dstack((npImage, npAlpha))
        
        print(npImage.shape)
        
        # Save with alpha
        Image.fromarray(npImage).save(f"output/frame_{frame_count:05d}.png", "PNG")
        #cv2.imwrite(f"output/frame_{frame_count:05d}.png", image)
        
        # read next image
        still_reading, image = video_capture.read()
        frame_count += 1
        
def make_gif(gif_path, frame_folder="output"):
    images = glob.glob(f"{frame_folder}/*.png")
    images.sort()
    frames = [Image.open(image) for image in images]
    frame_one = frames[0]
    frame_one.save(gif_path, format="GIF", append_images=frames,
                   save_all=True, duration=50, loop=0, transparency=0)
    
def main():
    layout = [
        [
            sg.Text("MP4 File"),
            sg.Input(size=(25, 1), key="-FILENAME-", disabled=True),
            sg.FileBrowse(file_types=file_types),
        ],
        [
            sg.Text("GIF File Save Location"),
            sg.Input(size=(25, 1), key="-OUTPUTFILE-", disabled=True),
            sg.SaveAs(file_types=file_types),
            
        ],
        [sg.Button("Convert to GIF")],
    ]
    window = sg.Window("MP4 to GIF Converter", layout)
    
    while True:
        event, values = window.read()
        mp4_path = values["-FILENAME-"]
        gif_path = values["-OUTPUTFILE-"]
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event in ["Convert to GIF"]:
            if mp4_path and gif_path:
                convert_mp4_to_jpgs(mp4_path)
                make_gif(gif_path)
                sg.popup(f"GIF created: {gif_path}")
    window.close()
    
if __name__ == "__main__":
    main()