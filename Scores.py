import tkinter
from tkinter import *
import matplotlib.pyplot as plt


lis = [0, 0, 0, 0, 0]
def Chart(Val):
    if plt.fignum_exists(1):
        plt.close()
        fig, ax = plt.subplots()
        number = lis
        lis[rb.get() - 1] += 1
        label = ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"]
        ax.bar(label, number)
        plt.yticks(number)
        plt.show()
    else:
        fig, ax = plt.subplots()
        lis[rb.get() - 1] += 1
        label = ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"]
        number = lis
        ax.bar(label, number)
        plt.yticks(number)
        plt.show()


tk = tkinter.Tk()
tk.geometry("500x500")
txt = "پروژه چقدر براتون جالب بود؟"
text = Label(tk, text=txt)
text.pack()
rb = IntVar()

rb1 = Radiobutton(tk, text="1 کمترین", variable=rb, value=1)
rb2 = Radiobutton(tk, text="2", variable=rb, value=2)
rb3 = Radiobutton(tk, text="3", variable=rb, value=3)
rb4 = Radiobutton(tk, text="4", variable=rb, value=4)
rb5 = Radiobutton(tk, text="5 بیشترین", variable=rb, value=5)
rb1.pack()
rb2.pack()
rb3.pack()
rb4.pack()
rb5.pack()
button = Button(tk, text='ثبت', command=lambda: Chart(rb))

button.pack()
tk.mainloop()


