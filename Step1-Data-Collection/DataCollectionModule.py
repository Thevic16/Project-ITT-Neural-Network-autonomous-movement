import pandas as pd
import os
import cv2
from datetime import datetime
from picamera.array import PiRGBArray
from picamera import PiCamera
import time


# Speech Recognition
#///////////////////////////////////////////////////////////////////////////////////////////
# Import threading module.
import threading

#Import serial module.
import serial

# Import Speech Recognition Module.
import argparse
import os
import queue
import sounddevice as sd
import vosk
import sys
import json

#Defining Queue that will be use in the Speech Recognition
q = queue.Queue()

# Set up serialPorts.
serialPortSTM32 = serial.Serial(port="/dev/ttyACM0", baudrate=9600,bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)

#Functions that will be use in the Speech Recognition module
def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))
    

#///////////////////////////////////////////////////////////////////////////////////////////

global imgList, directionList
countFolder = 0
count = 0
imgList = []
directionList = []


#GET CURRENT DIRECTORY PATH
myDirectory = os.path.join(os.getcwd(), 'DataCollected')
# print(myDirectory)


# CREATE A NEW FOLDER BASED ON THE PREVIOUS FOLDER COUNT
while os.path.exists(os.path.join(myDirectory,f'IMG{str(countFolder)}')):
        countFolder += 1
newPath = myDirectory +"/IMG"+str(countFolder)
os.makedirs(newPath)


# SAVE IMAGES IN THE FOLDER
def saveData(img,direction):
    global imgList, directionList
    now = datetime.now()
    timestamp = str(datetime.timestamp(now)).replace('.', '')
    #print("timestamp =", timestamp)
    fileName = os.path.join(newPath,f'Image_{timestamp}.jpg')
    cv2.imwrite(fileName, img)
    imgList.append(fileName)
    directionList.append(direction)
    
# SAVE LOG FILE WHEN THE SESSION ENDS
def saveLog():
    global imgList, directionList
    rawData = {'Image': imgList,
                'Direction': directionList}
    df = pd.DataFrame(rawData)
    df.to_csv(os.path.join(myDirectory,f'log_{str(countFolder)}.csv'), index=False, header=False)
    print('Log Saved')
    print('Total Images: ',len(imgList))
    
    
def saveDataAndLog(direction):
        # initialize the camera and grab a reference to the raw camera capture
        camera = PiCamera()
        camera.rotation = 90
        #camera.resolution = (900, 540)
        #camera.crop = (0.0, 0.0, 0.6, 0.95)
        rawCapture = PiRGBArray(camera)
        # allow the camera to warmup#
        time.sleep(2)
        # grab an image from the camera
        camera.capture(rawCapture, format="bgr")
        image = rawCapture.array
        image = image[500:1944,0:2592]
        #image = image[500:1744,400:2592] # Resolucion ajustada 15-12-2021
        camera.close()

        saveData(image, direction)
        cv2.waitKey(1)
        #cv2.imshow("Image", image)
    
#////////////////////////////////////////////////////////////////////////////////////////////

def speech_recognition_thread():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-l', '--list-devices', action='store_true',
        help='show list of audio devices and exit')
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])
    parser.add_argument(
        '-f', '--filename', type=str, metavar='FILENAME',
        help='audio file to store recording to')
    parser.add_argument(
        '-m', '--model', type=str, metavar='MODEL_PATH',
        help='Path to the model')
    parser.add_argument(
        '-d', '--device', type=int_or_str,
        help='input device (numeric ID or substring)')
    parser.add_argument(
        '-r', '--samplerate', type=int, help='sampling rate')
    args = parser.parse_args(remaining)

    try:
        if args.model is None:
            args.model = "model"
        if not os.path.exists(args.model):
            print ("Please download a model for your language from https://alphacephei.com/vosk/models")
            print ("and unpack as 'model' in the current folder.")
            parser.exit(0)
        if args.samplerate is None:
            device_info = sd.query_devices(args.device, 'input')
            # soundfile expects an int, sounddevice provides a float:
            args.samplerate = int(device_info['default_samplerate'])

        model = vosk.Model(args.model)

        if args.filename:
            dump_fn = open(args.filename, "wb")
        else:
            dump_fn = None

        with sd.RawInputStream(samplerate=args.samplerate, blocksize = 8000, device=args.device, dtype='int16',
                                channels=1, callback=callback):
                print('#' * 80)
                print('Press Ctrl+C to stop the recording')
                print('#' * 80)

                rec = vosk.KaldiRecognizer(model, args.samplerate)
                #Defining variable to compare
                text_str = ""
                past_text_str = ""
                while True:
                    data = q.get()
                    if rec.AcceptWaveform(data):
                        text_str = str(json.loads(rec.Result()+"")['text'])
                        #print("Mensaje voz: "+text_str)
                    else:
                        text_str = str(json.loads(rec.PartialResult()+"")['partial'])
                        #print("Mensaje voz: "+text_str)
                    if dump_fn is not None:
                        dump_fn.write(data)

        
                    if text_str != past_text_str:
                        # Conditions.
                        if "sofía" in text_str and "delante"  in text_str:
                            saveDataAndLog(0)
                            serialPortSTM32.write(b"w \r\n")
                            print("Comando de voz hacia adelante")
                            
                        elif "sofía" in text_str and "derecha" in text_str:
                            saveDataAndLog(3)
                            serialPortSTM32.write(b"d \r\n")
                            print("Comando de voz hacia la derecha")
            
                        elif "sofía" in text_str and "atrás" in text_str:
                            saveDataAndLog(1)
                            serialPortSTM32.write(b"s \r\n")
                            print("Comando de voz hacia atras")
                            
                        elif "sofía" in text_str and "izquierda" in text_str:
                            saveDataAndLog(2)
                            serialPortSTM32.write(b"a \r\n")
                            print("Comando de voz hacia la izquierda")
                        elif ("sofía" in text_str and "deten" in text_str) or ("sofía" in text_str and "párate" in text_str) or ("sofía" in text_str and "alto" in text_str):
                            saveDataAndLog(4)
                            saveLog()
                            return
                        
                    past_text_str = text_str


    except KeyboardInterrupt:
        print('\nDone')
        parser.exit(0)
    except Exception as e:
        parser.exit(type(e).__name__ + ': ' + str(e))
        print(e)


#////////////////////////////////////////////////////////////////////////////////////////////


if __name__ == '__main__':
    thread1 = threading.Thread(target=speech_recognition_thread)
    thread1.start()  


