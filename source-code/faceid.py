from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.logger import Logger
from kivy.core.text import LabelBase
from kivy.animation import Animation
from kivy.graphics import Color, Rectangle

import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

Window.clearcolor = (0.05, 0.05, 0.05, 1)  # Darker background

class CamApp(App):

    def build(self):
        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        self.web_cam = Image(size_hint=(1, .8), allow_stretch=True)

        # Background image for the layout
        with layout.canvas.before:
            Color(1, 1, 1, 1)  # Base color for transparency
            self.rect = Rectangle(size=layout.size, pos=layout.pos, source='background_gradient.png')

        self.button = Button(text="Verify", size_hint=(1, .1), font_size='20sp', background_normal='')
        self.button.background_color = (0.3, 0.3, 0.3, 1)  # Default color
        self.button.bind(on_press=self.animate_button)

        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1, .1), color=(0, 1, 0, 1))
        self.welcome_label = Label(text="", size_hint=(1, .1), color=(0, 1, 0, 1))  # For welcome message

        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)
        layout.add_widget(self.welcome_label)

        self.model = tf.keras.models.load_model('siamesemodel-aug.h5', custom_objects={'L1Dist': L1Dist})
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return layout

    def animate_button(self, instance):
        anim = Animation(background_color=(0.2, 0.5, 0.8, 1), duration=0.3)
        anim += Animation(background_color=(0.3, 0.3, 0.3, 1), duration=0.3)
        anim.start(instance)
        self.verify()

    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, (100, 100))
        img = img / 255.0
        return img

    def update(self, *args):
        ret, frame = self.capture.read()
        frame = cv2.resize(frame, (640, 360))
        frame = frame[120:120 + 250, 200:200 + 250, :]
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def verify(self, *args):
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = cv2.resize(frame, (640, 360))
        frame = frame[50:50 + 250, 200:200 + 250, :]
        cv2.imwrite(SAVE_PATH, frame)

        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)

        detection = np.sum(np.array(results) > 0.5)
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        verified = verification > 0.5

        # Update verification label and set welcome message
        if verified:
            self.verification_label.text = 'Verified'
            self.welcome_label.text = "Welcome to the system!"
            Logger.info("User verified and welcomed.")
        else:
            self.verification_label.text = 'Unverified'
            self.welcome_label.text = "Authentication Failure. Please Try Again"
            Logger.info("User not verified.")

        return results, verified

if __name__ == '__main__':
    CamApp().run()
