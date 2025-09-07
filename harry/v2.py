import tkinter as tk
from tkinter import simpledialog, messagebox
from palm_system import PalmRecognizer
import threading

class PalmApp:
    def __init__(self, root):
        self.root = root
        root.title("Palm Recognition")
        root.geometry("360x280")
        self.sys = PalmRecognizer()

        tk.Button(root, text="Enroll User", width=25, command=self.enroll_ui).pack(pady=10)
        tk.Button(root, text="Start Recognition", width=25, command=self.start_recognize_thread).pack(pady=10)
        tk.Button(root, text="Quit", width=25, fg="white", bg="firebrick", command=self.quit_app).pack(pady=12)
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(root, textvariable=self.status_var, fg="blue").pack(pady=6)

    def enroll_ui(self):
        user_id = simpledialog.askstring("Enroll", "Enter unique user ID:", parent=self.root)
        if not user_id: return
        self.status_var.set(f"Enrolling {user_id} ...")
        threading.Thread(target=self._enroll_thread, args=(user_id,), daemon=True).start()

    def _enroll_thread(self, user_id):
        self.sys.enroll_user(user_id)
        self.status_var.set(f"Enrolled {user_id}")

    def start_recognize_thread(self):
        if not self.sys.db:
            messagebox.showwarning("No Users", "Enroll first!")
            return
        threading.Thread(target=self.sys.recognize, daemon=True).start()

    def quit_app(self):
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = PalmApp(root)
    root.mainloop()
