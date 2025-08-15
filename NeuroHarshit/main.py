import customtkinter as ctk
from PIL import ImageTk


class MainWindow(ctk.CTk):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.basic_structure()


    def basic_structure(self) -> None:
        # setting appearance
        ctk.set_appearance_mode('dark')
        ctk.set_default_color_theme('Utils/theme.json')
        self.geometry('700x500')
        self.minsize(700, 500)
        self.title('NueroHarshit')

        # setting icon
        self.imagepath = ImageTk.PhotoImage(file= 'Icons/app.png')
        self.wm_iconbitmap()
        self.iconphoto(True, self.imagepath)


if __name__ == '__main__':
    app = MainWindow()
    app.mainloop()
