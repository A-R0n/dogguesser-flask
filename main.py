import cv2
import time
import urllib
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
from flask import Flask, render_template, Response, request
import os
from capture_image import captureImage
from time import perf_counter
# from guppy import hpy
import ssl

app = Flask(__name__)
global capture, switch, epoch_time,default_config
epoch_time = None
capture = 0
switch=1
default_config = {
    'display_video_feed': 'flex',
    'display_captured_img': 'none',
    'epoch_time': epoch_time,
    'imageCapturedSuccess': False
}

imagenet_classes_file = "imagenet_classes.txt"
imagenet_classes_download = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
BGR_WHITE_COLOR = (255, 255, 255)

@app.route('/')
def index():
    global default_config
    return render_template('index.html', data=default_config)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def get_output(model, tensor):
    with torch.no_grad():
        return model(tensor)
    
def get_imagenet_classes():
    ## if we don't already have it
    if not os.path.isfile(imagenet_classes_file):
        print(f'we need to download the imagenet clases from github')
        urllib.request.urlretrieve(imagenet_classes_download, imagenet_classes_file)
    else:
        print(f'imagenet classes available to us locally')

def structure_imagenet_classes(b):
    with open(b, "r") as f:
        return [s.strip() for s in f.readlines()] 
    
def get_most_probable(probabilities, n):
    return torch.topk(probabilities, n)

def display_results(topn_prob, topn_catid, categories):
    for i in range(topn_prob.size(0)):
        print(f'topn_catid {topn_catid}')
        print(f'topn_catid at index {topn_catid[i]}')
        try:
            idx_category = topn_catid[i]
            print(f'idx category {idx_category}')
        except IndexError:
            continue
        if idx_category >= 151 and idx_category <= 275:
            print(f'we found a dog')
            try:
                category_name = categories[idx_category]
                prob = str(round(float(topn_prob[i].item()*100), 1)) + "%"
                print(category_name, prob)
            except IndexError:
                continue
        else:
            break
    try:
        idx_category = topn_catid[i]
        print(f'idx category {idx_category}')
        if idx_category >= 151 and idx_category <= 275:
            return str(categories[topn_catid[0]]).title()
        else:
            return 'N/A'
    except IndexError:
        return "Index Error"
    
def does_blurred_image_exist() -> bool:
    global epoch_time
    blurred_image_fp = f'static/output/capture_frame_blur_{epoch_time}.png'
    if os.path.isfile(blurred_image_fp):
        return True
    return False

def blur_frame(frame):
    return cv2.blur(frame, (100,100))

def put_text_on_frame(frame, top_guess):
    return cv2.putText(frame, top_guess, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, BGR_WHITE_COLOR, 3, cv2.LINE_AA) 

def guess_dog_breed(p):
    model = timm.create_model("inception_v3", pretrained=True)
    model.eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    img = Image.open(p).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    output = get_output(model, tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    get_imagenet_classes()
    categories = structure_imagenet_classes("imagenet_classes.txt")
    topn_prob, topn_catid = get_most_probable(probabilities, 3)
    return display_results(topn_prob, topn_catid, categories)

def save_frame(p, frame):
    cv2.imwrite(p, frame)

def path_current_frame(epoch_time):
    return os.path.sep.join(['static', "output/capture_frame_analysis_{}.png".format(str(epoch_time).replace(":",''))])

def gen_frames():
    global capture, epoch_time
    camera = cv2.VideoCapture(0)  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            if capture == 1:
                start = perf_counter()
                current_frame_path = path_current_frame(epoch_time)
                save_frame(current_frame_path, frame)
                dog_breed_guessed = guess_dog_breed(current_frame_path)

                capturedImageBlurredWithText = captureImage(frame, epoch_time)
                blurred_img = capturedImageBlurredWithText._blurred()
                frame_blurred_with_text = capturedImageBlurredWithText._withText(dog_breed_guessed, blurred_img)
                capturedImageBlurredWithText._save(frame_blurred_with_text)

                end = perf_counter()
                total = end - start
                print(f'total time to guess dog: {total} seconds!')
                # capture = 0

            try:
                ## instead of if not capture
                ## it should be if the captured image doesn't exist
                if not capture:
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f'exception when yielding frames : {e}')
                pass

def update_config(epoch_time):
    return {
        'display_video_feed': 'none',
        'display_captured_img': 'flex',
        'epoch_time': epoch_time,
        'imageCapturedSuccess': True
    }

def get_file_size():
    global epoch_time
    file_stats = os.stat(f'static/output/capture_frame_blur_{epoch_time}.png')
    return file_stats.st_size

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera,default_config,epoch_time

    if request.method == 'POST':
        if request.form.get('click') == 'guess':
            global capture, epoch_time, default_config
            epoch_time = time.time_ns()
            capture=1
            while capture == 1:
                if does_blurred_image_exist():
                    file_size = get_file_size()
                    print(f'file_size: {file_size}')
                    ## we still have to wait cus itll find half the image to exist and render that
                    if file_size > 0:
                        ## it still might not be fully loaded even though a number is finally available to us
                        time.sleep(.05)
                        capture = 0
                        break

            updated_config = update_config(epoch_time)
            return render_template('index.html', data=updated_config)
        elif request.form.get('dog') == 'remove':
            print(f'removing dog')
            path_analysis = path_current_frame(epoch_time)
            path_blurred = f'static/output/capture_frame_blur_{epoch_time}.png'
            ps = [path_analysis, path_blurred]
            for p in ps:
                print(f'removing p {p}')
                os.remove(p)
            print(f'files removed from output...')
            # h=hpy()
            # print(f'app usage...')
            # print(h.heap())
            # time.sleep(10)
            # return render_template('index.html', data=default_config)
    elif request.method=='GET':
        return render_template('index.html', data=default_config)

    return render_template('index.html', data=default_config)

if __name__ == '__main__':
    ## debug true lets us update our code without restarting the server
    ## in prod, we can't do this
    app.run(debug=True)
    # app.run()
    # app.run(ssl_context='adhoc')
    # context = ssl.SSLContext()
    # context.load_cert_chain('fullchain.pem', 'privkey.pem')
    # app.run(ssl_context="adhoc", debug=True)

    # app.run(debug=False)