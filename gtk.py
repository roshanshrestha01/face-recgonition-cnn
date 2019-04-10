import os
from builtins import ord

import gi
import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torchvision import transforms
from torch.autograd import Variable

from dataloaders import capture_dataloader
from networks import NNetwork, CNNetwork
from settings import HAAR_CASCADE, CAPTURE_DIR, ORL_TRAINED_MODEL, USE_CNN, RESIZE
from utils import check_folder

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

face_cascade = cv2.CascadeClassifier(HAAR_CASCADE)


class FaceRecognitionWindow(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title="Face Recognition")
        self.model = None
        try:
            self.classes = capture_dataloader.dataset.classes
        except:
            pass
        self.set_position(Gtk.WindowPosition.CENTER)
        self.set_border_width(10)
        self.CITY_NAME = None  # for refresh action
        self.gui = self.setup()
        self.add(self.gui)

    def setup(self):
        box_outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        listbox = Gtk.ListBox()
        listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        box_outer.pack_start(listbox, True, True, 0)

        row = Gtk.ListBoxRow()
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=50)
        row.add(hbox)
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        hbox.pack_start(vbox, True, True, 0)

        subject_name = Gtk.Label(xalign=0)
        subject_name.set_text("Subject name")
        self.subject_name = Gtk.Entry()
        self.subject_name.set_size_request(400, 20)
        self.subject_name.set_text("S1")
        self.subject_name.set_width_chars(40)

        vbox.pack_start(subject_name, False, False, 0)
        vbox.pack_start(self.subject_name, False, False, 0)

        listbox.add(row)

        row = Gtk.ListBoxRow()
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=50)
        row.add(hbox)
        self.capture_image = Gtk.Button.new_with_label("Capture Image")
        self.capture_image.connect("clicked", self.open_capture_image_window)
        self.capture_image.set_size_request(200, 20)

        self.sort_image = Gtk.Button.new_with_label("Sort Image")
        self.sort_image.connect("clicked", self.sort_images)
        self.sort_image.set_size_request(200, 20)

        hbox.pack_end(self.sort_image, True, True, 0)
        hbox.pack_end(self.capture_image, True, True, 0)

        listbox.add(row)

        row = Gtk.ListBoxRow()
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=50)
        row.add(hbox)
        self.traing_model = Gtk.Button.new_with_label("Train Model")
        self.traing_model.connect("clicked", self.training_model)
        self.traing_model.set_size_request(200, 20)

        self.predict_model = Gtk.Button.new_with_label("Predict Video")
        self.predict_model.connect("clicked", self.open_predict_window)
        self.predict_model.set_size_request(200, 20)

        hbox.pack_end(self.predict_model, True, True, 0)
        hbox.pack_end(self.traing_model, True, True, 0)

        listbox.add(row)

        return box_outer

    def open_capture_image_window(self, button):
        subject_name = self.subject_name.get_text()

        cap = cv2.VideoCapture(2)
        count = 1

        while True:
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            roi_gray = None
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
                # if roi_gray is not None:
                #     check_folder(CAPTURE_DIR)
                #     subject_root = os.path.join(CAPTURE_DIR, subject_name)
                #     check_folder(subject_root)
                #     cv2.imwrite(os.path.join(subject_root, '{}.jpg'.format(count)), roi_gray)
                #     count += 1
                #     print(count)

            cv2.imshow('Video', img)
            k = cv2.waitKey(30) & 0xff
            if k == ord('c'):
                check_folder(CAPTURE_DIR)
                subject_root = os.path.join(CAPTURE_DIR, subject_name)
                check_folder(subject_root)
                count = len(os.listdir(subject_root))
                if roi_gray is not None:
                    cv2.imwrite(os.path.join(subject_root, '{}.jpg'.format(count)), roi_gray)
                    count += 1
                else:
                    print('ROI cannot be found.')
            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def sort_images(self, button):
        pass

    def training_model(self, button):
        model = CNNetwork() if USE_CNN else NNetwork()
        state_dict = torch.load(ORL_TRAINED_MODEL)
        model.load_state_dict(state_dict)
        for param in model.parameters():
            param.requires_grad = False
        model.fc2 = nn.Linear(1024, len(self.classes))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        epochs = 10
        steps = 0
        running_loss = 0
        print_every = 5
        for epoch in range(epochs):
            for inputs, labels in capture_dataloader:
                steps += 1
                # Move input and label tensors to the default device

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in capture_dataloader:
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch + 1}/{epochs}.. "
                          f"Train loss: {running_loss / print_every:.3f}.. "
                          f"Test loss: {test_loss / len(capture_dataloader):.3f}.. "
                          f"Test accuracy: {accuracy / len(capture_dataloader):.3f}")
                    running_loss = 0
                    model.train()
        self.model = model

        dialog = Gtk.MessageDialog(self, 0, Gtk.MessageType.INFO, Gtk.ButtonsType.OK, "Training Finished.")
        dialog.format_secondary_text(
            "Press ok to continue.")
        dialog.run()
        dialog.destroy()
        pass

    def convert_to_pil(self, array):
        return Image.fromarray(array, 'L')

    def open_predict_window(self, button):
        cap = cv2.VideoCapture(2)
        count = 1

        while True:
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            roi_gray = None
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
                if roi_gray is not None:
                    image = self.convert_to_pil(roi_gray)
                    test_transforms = transforms.Compose([
                        transforms.Scale(RESIZE),
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x * 255),
                    ])
                    tensor = test_transforms(image)
                    image_tensor = tensor.unsqueeze_(0)
                    input = Variable(image_tensor)
                    output = self.model(input)
                    ps = torch.exp(output)
                    top_p, top_class = ps.topk(1, dim=1)
                    index = top_class.reshape(-1)[0]
                    subject_name = self.classes[index]
                    cv2.putText(img, subject_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 255, 0), 2)
                    print(subject_name)
            cv2.imshow('Video', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


win = FaceRecognitionWindow()
win.connect("destroy", Gtk.main_quit)
win.show_all()
Gtk.main()
