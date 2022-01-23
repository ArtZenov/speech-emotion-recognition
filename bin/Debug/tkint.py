import tkinter as tk
import livetesting as lt


def btn():
    label2['text'] = 'Please talk'
    emotion, text, data = lt.main()
    label1['text'] = f'{emotion}'
    label2['text'] = f'{text}'
    lt.plt.plot(data)
    lt.plt.show()
    lt.plt.savefig('wave.png')


window = tk.Tk()
window.title("Звук")
window.geometry("300x300")

label1 = tk.Label(width='290', height='3', bg='black', foreground='white')
label2 = tk.Label(width='290', height='3', bg='black', foreground='white')
button = tk.Button(text='Execute', width='15', height='3', bg='grey', command=btn)

label1.pack(padx=10, pady=10)
label2.pack(padx=10, pady=10)
button.pack(padx=10, pady=10)

window.mainloop()
