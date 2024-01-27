import cv2
import time
import urllib
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
import sys

def main():
    Main()

class Main:

    def __init__(self):
        self.window_name = 'test'
        self.cam = cv2.VideoCapture(0)
        self.model_type = 'inception_v3'
        self.is_pretrained = True
        self.default_config = {}
        self.imagenet_classes_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        self.imagenet_classes_filename = "imagenet_classes.txt"
        self.n = 3
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.org = (200, 200) 
        self.fontScale = 3
        self.bgr_white_color = (255, 255, 255)
        self.bgr_black_color = (0, 0, 0)
        self.thickness = 3
        self.output_dir = 'output/'  
        self._main()

    def _guess_dog_breed(self):
        self.model = self._create_model()
        self._eval_model()
        self.config = self._create_config()
        self.transform = create_transform(**self.config)
        self.img = self._open_and_convert_img_rgb()
        self.tensor = self._create_tensor()
        self.output = self._get_output()
        self.probabilities = self._compute_probabilities()
        self._get_imagenet_classes()
        self.categories = self._structure_imagenet_classes()
        self.topn_prob, self.topn_catid = self._get_most_probable()
        self._display_results()

    def _get_imagenet_classes(self):
        ## imports a list of all the different classes (1000 in total) the CNN is trained upon
        urllib.request.urlretrieve(self.imagenet_classes_url, self.imagenet_classes_filename)

    def _structure_imagenet_classes(self):
        with open(self.imagenet_classes_filename, "r") as f:
            return [s.strip() for s in f.readlines()]    

    def _get_most_probable(self):
        return torch.topk(self.probabilities, self.n)
    
    def _display_results(self):
        for i in range(self.topn_prob.size(0)):
            category_name = self.categories[self.topn_catid[i]]
            prob = str(round(float(self.topn_prob[i].item()*100), 1)) + "%"
            print(category_name, prob)
        self.top_guess = str(self.categories[self.topn_catid[0]]).title()

    def _open_and_convert_img_rgb(self):
        return Image.open(self.img_name).convert('RGB')
    
    def _create_tensor(self):
        return self.transform(self.img).unsqueeze(0)

    def _compute_probabilities(self):
        return torch.nn.functional.softmax(self.output[0], dim=0)

    def _get_output(self):
        with torch.no_grad():
            return self.model(self.tensor)

    def _create_config(self):
        try:
            return resolve_data_config(self.default_config, model=self.model)
        except AssertionError:
            print('First, we need to create our pyTorch IMage Model (TIMM)!')
            sys.exit()
        
    def _create_model(self):
        return timm.create_model(self.model_type, pretrained=self.is_pretrained)
    
    def _eval_model(self):
        return self.model.eval()

    def _space_key_pressed(self) -> bool:
        if self.k%256 == 32:
            return True
        return False
    
    def _esc_key_pressed(self) -> bool:
        ## need to map the space key to the Idexx image icon
        if self.k%256 == 27:
            return True
        return False
    
    def _capture_img(self):
        cv2.imwrite(self.img_name, self.frame)

    def _pause_video(self):
        cv2.waitKey(-1)

    def _get_epoch_nano(self):
        return time.time_ns()
    
    def _blur_frame(self):
        self.kernal_size = (100,100)
        self.blurred_frame = cv2.blur(self.frame, self.kernal_size)
        cv2.imshow(self.window_name, self.blurred_frame)

    def _put_text_on_blurred_frame(self):
        return cv2.putText(
            self.blurred_frame,
            self.top_guess,
            self.org, self.font,
            self.fontScale,
            self.bgr_white_color,
            self.thickness,
            cv2.LINE_AA
        ) 

    def _display_top_guess(self):
        self.blurred_frame_with_text = self._put_text_on_blurred_frame()
        cv2.imshow(self.window_name, self.blurred_frame_with_text)  

    def _main(self):
        cv2.namedWindow(self.window_name)
        while True:
            self.ret, self.frame = self.cam.read()
            if not self.ret:
                print("failed to grab frame")
                break
            cv2.imshow(self.window_name, self.frame)
            self.k = cv2.waitKey(1)
            if self._esc_key_pressed():
                break
            elif self._space_key_pressed():
                self.epoch_nano = self._get_epoch_nano()
                self.img_name = f'{self.output_dir }{self.epoch_nano}.jpg'
                self._capture_img()
                self._guess_dog_breed()
                self._blur_frame()
                self._display_top_guess()
                self._pause_video()

        self.cam.release()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()