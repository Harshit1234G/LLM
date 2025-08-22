import customtkinter as ctk
from PIL import ImageTk
from dotenv import load_dotenv
from Agent.chatbot import ChatBot


class MainWindow(ctk.CTk):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        load_dotenv()

        self.chat_bot = ChatBot(vector_db_path= r'.\Databases\faiss_index')
        self.basic_structure()
        self.input_field()


    def basic_structure(self) -> None:
        width = 700
        height = 500

        # setting appearance
        ctk.set_appearance_mode('dark')
        self.geometry(f'{width}x{height}')
        self.minsize(width, height)
        self.title('NeuroHarshit')
        self.configure(bg= '#212121')

        # setting icon
        self.imagepath = ImageTk.PhotoImage(file= 'Icons/app.png')
        self.wm_iconbitmap()
        self.iconphoto(True, self.imagepath)


    def input_field(self) -> None:
        self.input = ctk.CTkTextbox(
            master= self, 
            height= 100,
            fg_color= '#303030',
            corner_radius= 20
        )
        self.input.pack(
            fill= 'x',
            side= 'bottom',
            anchor= 's',
            expand= True,
            padx= 100,
            pady= 10
        )


if __name__ == '__main__':
    app = MainWindow()
    app.mainloop()
