import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk


class FaceRecognitionWindow(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title="Face Recognition")
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
        print(self.subject_name.get_text())
        pass

    def sort_images(self, button):
        pass

    def training_model(self, button):
        pass

    def open_predict_window(self, button):
        pass


win = FaceRecognitionWindow()
win.connect("destroy", Gtk.main_quit)
win.show_all()
Gtk.main()
