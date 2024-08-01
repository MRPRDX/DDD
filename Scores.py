import tkinter
from tkinter import *
import matplotlib.pyplot as plt


lis = [0, 0, 0, 0, 0]

def lispercent():
    r = 0
    percent = []
    for i in lis:
        r += i
    for j in lis:
       percent.append(int(j/r*100))
    return percent
def Chart(Val):
    if plt.fignum_exists(1):
        plt.close()
        number = lis
        lis[rb.get() - 1] += 1
        label = ["1 Star " + str(lispercent()[0])+"%", "2 Stars " + str(lispercent()[1])+"%",
                 "3 Stars " + str(lispercent()[2])+"%", "4 Stars " + str(lispercent()[3])+"%",
                 "5 Stars " + str(lispercent()[4])+"%"]
        plt.pie(number)
        plt.legend(labels=label)
        plt.show()
    else:
        lis[rb.get() - 1] += 1
        label = ["1 Star "+str(lispercent()[0])+"%", "2 Stars "+str(lispercent()[1])+"%",
                 "3 Stars "+str(lispercent()[2])+"%", "4 Stars "+str(lispercent()[3])+"%",
                 "5 Stars "+str(lispercent()[4])+"%"]
        number = lis
        plt.pie(number)
        plt.legend(labels=label)
        plt.show()


tk = tkinter.Tk()
tk.geometry("500x500")
txt = "پروژه چقدر براتون جالب بود؟"
text = Label(tk, text=txt, font="bnazanin 25")
text.pack()
rb = IntVar()

rb1 = Radiobutton(tk, text="1 کمترین", variable=rb, value=1, font="bnazanin 25")
rb2 = Radiobutton(tk, text="2", variable=rb, value=2, font="bnazanin 25")
rb3 = Radiobutton(tk, text="3", variable=rb, value=3, font="bnazanin 25")
rb4 = Radiobutton(tk, text="4", variable=rb, value=4, font="bnazanin 25")
rb5 = Radiobutton(tk, text="5 بیشترین", variable=rb, value=5, font="bnazanin 25")
rb1.pack()
rb2.pack()
rb3.pack()
rb4.pack()
rb5.pack()
button = Button(tk, text='ثبت', command=lambda: Chart(rb), font="bnazanin 25")
text2 = Label(tk, text="هوش مصنوعی گروه تشخیص خواب آلودگی", font="bnazanin 15")
button.pack()
text2.pack()
tk.mainloop()


