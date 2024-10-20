import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox
from PIL import Image, ImageDraw, ImageOps, ImageTk
import cv2
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Combine train and test sets
X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))

# Reshape and normalize the data
X = X.reshape((X.shape[0], -1)).astype('float32') / 255.0

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train the KNN model with user-defined k
k = 3  
knn = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1)
knn.fit(X_train, y_train)

# Evaluate the model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

# Print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot accuracy
def plot_accuracy(accuracies):
    plt.plot(accuracies)
    plt.title('Model Accuracy over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.show()

# GUI Application
class DrawApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Draw a Number")

        # Set window size
        self.window_width = 1024
        self.window_height = 576
        self.master.geometry(f"{self.window_width}x{self.window_height}")

        # Load and set background image
        self.background_image = Image.open(r"D:\CODE\python\a.png")
        self.background_image = self.background_image.resize((self.window_width, self.window_height), Image.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(self.background_image)

        # Create a label for the background image
        self.bg_label = tk.Label(self.master, image=self.bg_photo)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Create a canvas for drawing
        self.canvas_width = 280
        self.canvas_height = 280
        self.drawing_canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.drawing_canvas.place(relx=0.3, rely=0.55, anchor=tk.CENTER)

        # Create a label to display predictions
        self.prediction_label = tk.Label(master, text="", font=("Arial", 16))
        self.prediction_label.place(relx=0.3, rely=0.85, anchor=tk.CENTER)

        # Create a frame to hold the buttons
        self.button_frame = tk.Frame(master)
        self.button_frame.place(relx=0.3, rely=0.9, anchor=tk.CENTER)

        self.predict_button = tk.Button(self.button_frame, text="Dự đoán", command=self.predict)
        self.predict_button.grid(row=0, column=0, padx=5, pady=5)

        self.clear_button = tk.Button(self.button_frame, text="Xóa", command=self.clear)
        self.clear_button.grid(row=0, column=1, padx=5, pady=5)

        self.color_button = tk.Button(self.button_frame, text="Chọn Màu", command=self.choose_color)
        self.color_button.grid(row=0, column=2, padx=5, pady=5)

        self.save_button = tk.Button(self.button_frame, text="Lưu Ảnh", command=self.save_image)
        self.save_button.grid(row=0, column=3, padx=5, pady=5)

        self.upload_button = tk.Button(self.button_frame, text="Tải Ảnh", command=self.upload_image)
        self.upload_button.grid(row=0, column=4, padx=5, pady=5)

        self.exit_button = tk.Button(self.button_frame, text="Thoát", command=self.master.quit)
        self.exit_button.grid(row=0, column=5, padx=5, pady=5)

        # Help button
        self.help_button = tk.Button(self.button_frame, text="Hướng dẫn", command=self.show_help)
        self.help_button.grid(row=0, column=6, padx=5, pady=5)

        # Initialize image for drawing
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image)

        # Initialize history and predictions
        self.prediction_history = []  # To store prediction results

        # Bind paint function to mouse motion
        self.drawing_canvas.bind("<B1-Motion>", self.paint)

        self.r = 18  # Brush size
        self.current_color = "black"  # Default color

        self.brush_size_label = tk.Label(master, text="Kích thước bút:")
        self.brush_size_label.place(relx=0.3, rely=0.95, anchor=tk.CENTER)
        
        self.brush_size_scale = tk.Scale(master, from_=1, to=30, orient=tk.HORIZONTAL, command=self.update_brush_size)
        self.brush_size_scale.set(18)
        self.brush_size_scale.place(relx=0.3, rely=0.98, anchor=tk.CENTER)
        
    def choose_color(self):
        color = colorchooser.askcolor()[1]
        if color:
            self.current_color = color

    def paint(self, event):
        x, y = event.x, event.y
        self.drawing_canvas.create_oval(x - self.r, y - self.r, x + self.r, y + self.r, fill=self.current_color, outline=self.current_color)
        self.draw.ellipse([x - self.r, y - self.r, x + self.r, y + self.r], fill=0)

    def show_help(self):
        help_message = (
            "Hướng dẫn sử dụng:\n"
            "1. Vẽ số từ 0 đến 9 trên khung vẽ.\n"
            "2. Nhấn nút 'Dự đoán' để xem kết quả dự đoán của mô hình.\n"
            "3. Nhấn nút 'Xóa' để xóa khung vẽ và bắt đầu lại.\n"
            "4. Nhấn nút 'Chọn Màu' để thay đổi màu bút vẽ.\n"
            "5. Nhấn nút 'Lưu Ảnh' để lưu ảnh đã vẽ.\n"
            "6. Nhấn nút 'Tải Ảnh' để tải ảnh từ máy tính.\n"
            "7. Nhấn nút 'Thoát' để đóng ứng dụng."
        )
        messagebox.showinfo("Hướng dẫn", help_message)

    def clear(self):
        self.drawing_canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="")
        self.prediction_history = []

    def preprocess_image(self):
        img = self.image.copy()
        img = ImageOps.invert(img)
        img = img.convert('L')
        img = img.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img)

        _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(img_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)

            digit = img_array[y:y+h, x:x+w]
            square_size = max(w, h) + 4
            background = np.zeros((square_size, square_size), dtype=np.uint8)
            offset_x, offset_y = (square_size - w) // 2, (square_size - h) // 2
            background[offset_y:offset_y+h, offset_x:offset_x+w] = digit

            img_array = cv2.resize(background, (28, 28), interpolation=cv2.INTER_AREA)

        return img_array.reshape(1, -1).astype('float32') / 255.0

    def predict(self):
        img_array = self.preprocess_image()
        prediction = knn.predict(img_array)
        probabilities = knn.predict_proba(img_array)
        predicted_class = int(prediction[0])
        probability = probabilities[0][predicted_class]
        
        self.prediction_history.append((predicted_class, probability))
        self.prediction_label.config(text=f'Dự đoán: {predicted_class} (Xác suất: {probability:.2f})')

       # messagebox.showinfo("Kết quả dự đoán", f'Dự đoán: {predicted_class}\nXác suất: {probability:.2f}')

    def save_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg")])
        if file_path:
            self.image.save(file_path)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image = Image.open(file_path).convert("L")
            self.image = self.image.resize((self.canvas_width, self.canvas_height), Image.LANCZOS)
            self.draw = ImageDraw.Draw(self.image)
            self.clear()

    def update_brush_size(self, value):
        self.r = int(value)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()
