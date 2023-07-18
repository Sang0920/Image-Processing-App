import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

global i_img
global o_img
window = tk.Tk()
window.title("Image Processing")

# Create canvas to display images
canvas_i = tk.Canvas(window, width=600, height=400)
canvas_o = tk.Canvas(window, width=600, height=400)

label = tk.Label(window, text="Image Processing", font=("Arial Bold", 20))

def select_image():
    global i_img
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    i_img = cv2.imread(file_path)
    print(i_img.shape)
    if file_path:
        display_image(file_path)

def display_image(file_path):
    # Open and display original image
    img = Image.open(file_path)
    img.thumbnail((600, 400))
    img_tk = ImageTk.PhotoImage(img)
    canvas_i.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas_i.image = img_tk  # Store the image reference

def save_image():
    global o_img
    if o_img is not None:
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg")
        if save_path:
            cv2.imwrite(save_path, o_img)
            print("Image saved successfully.")

def exit():
    window.destroy()

def show_histogram():
    global i_img
    if i_img is not None:
        img_arr = cv2.cvtColor(i_img, cv2.COLOR_BGR2RGB)
        plt.hist(img_arr.ravel(), bins=256, color='gray')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.title('Histogram')
        plt.show()

def gaussian_filter():
    global i_img, o_img
    if i_img is not None:
        processed_img = cv2.GaussianBlur(i_img, (5, 5), 0)
        print(processed_img.shape)
        o_img = processed_img.copy()
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        # display processed image
        img = Image.fromarray(processed_img)
        img.thumbnail((600, 400))
        img_tk = ImageTk.PhotoImage(img)
        canvas_o.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas_o.image = img_tk  # Store the image reference
    else:
        # show error message
        message = "Please select an image first!"
        tk.messagebox.showerror(title="Error", message=message)

def quantize_image():
    global i_img, o_img
    if i_img is not None:
        img_arr = cv2.cvtColor(i_img, cv2.COLOR_BGR2RGB)
        img_arr = img_arr.astype(np.float32)
        img_arr /= 255.0
        img_arr = np.power(img_arr, 0.5)
        img_arr *= 255.0
        img_arr = img_arr.astype(np.uint8)
        o_img = img_arr.copy()
        # display processed image
        img = Image.fromarray(img_arr)
        img.thumbnail((600, 400))
        img_tk = ImageTk.PhotoImage(img)
        canvas_o.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas_o.image = img_tk  # Store the image reference
    else:
        # show error message
        message = "Please select an image first!"
        tk.messagebox.showerror(title="Error", message=message)

def morphological_operation():
    global i_img, o_img
    try:
        if i_img is None:
            pass
    except NameError:
        # show error message
        message = "Please select an image first!"
        tk.messagebox.showerror(title="Error", message=message)
    operation, kernel_type, kernel_size = None, None, None

    # show popup window to get operation type
    operation_types = [cv2.MORPH_OPEN, cv2.MORPH_CLOSE, cv2.MORPH_GRADIENT]
    operation_names = ["Open", "Close", "Gradient"]
    selected_operation = tk.StringVar()

    def on_operation_select():
        nonlocal operation
        operation = operation_types[int(selected_operation.get())]
        operation_window.destroy()

    operation_window = tk.Toplevel(window)
    operation_window.title("Select Operation")
    operation_label = tk.Label(operation_window, text="Select the morphological operation:")
    operation_label.pack()
    for i, op_name in enumerate(operation_names):
        rb = tk.Radiobutton(operation_window, text=op_name, variable=selected_operation, value=i)
        rb.pack(anchor=tk.W)
    confirm_button = tk.Button(operation_window, text="OK", command=on_operation_select)
    confirm_button.pack()
    operation_window.wait_window()

    # show popup window to get kernel size
    def on_kernel_size_submit():
        nonlocal kernel_size
        size_input = kernel_size_entry.get()
        x, y = map(int, size_input.split(','))
        kernel_size = (x, y)
        kernel_size_window.destroy()

    kernel_size_window = tk.Toplevel(window)
    kernel_size_window.title("Kernel Size")
    kernel_size_label = tk.Label(kernel_size_window, text="Enter the kernel size (odd value) (e.g. '5,5'):")
    kernel_size_label.pack()
    kernel_size_entry = tk.Entry(kernel_size_window)
    kernel_size_entry.pack()
    confirm_button = tk.Button(kernel_size_window, text="OK", command=on_kernel_size_submit)
    confirm_button.pack()
    kernel_size_window.wait_window()

    # show popup window to get kernel type
    kernel_types = [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]
    kernel_type_names = ["Rectangle", "Cross", "Ellipse"]
    selected_kernel_type = tk.StringVar()

    def on_kernel_type_select():
        nonlocal kernel_type
        operation = operation_types[int(selected_operation.get())]
        kernel_type_window.destroy()

    kernel_type_window = tk.Toplevel(window)
    kernel_type_window.title("Kernel Type")
    kernel_type_label = tk.Label(kernel_type_window, text="Select the kernel type:")
    kernel_type_label.pack()
    for i, type_name in enumerate(kernel_type_names):
        rb = tk.Radiobutton(kernel_type_window, text=type_name, variable=selected_kernel_type, value=i)
        rb.pack(anchor=tk.W)
    confirm_button = tk.Button(kernel_type_window, text="OK", command=on_kernel_type_select)
    confirm_button.pack()
    kernel_type_window.wait_window()

    if i_img is not None:
        if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
            # show error message
            message = "Kernel size should be odd!"
            tk.messagebox.showerror(title="Error", message=message)
        else:
            kernel = cv2.getStructuringElement(kernel_type, kernel_size)
            processed_img = cv2.morphologyEx(i_img, operation, kernel)
            o_img = processed_img.copy()
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            # display processed image
            img = Image.fromarray(processed_img)
            img.thumbnail((600, 400))
            img_tk = ImageTk.PhotoImage(img)
            canvas_o.create_image(0, 0, anchor=tk.NW, image=img_tk)
            canvas_o.image = img_tk
            
def k_means_pp():
    global i_img, o_img
    k = None
    try:
        if i_img is not None:
            Z = i_img.reshape(-1, 1)
            Z = np.float32(Z)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            
            # show pop-up window to get k value
            def on_k_submit():
                nonlocal k
                k = int(k_entry.get())
                k_window.destroy()

            k_window = tk.Toplevel(window)
            k_window.title("K-Means++")
            k_label = tk.Label(k_window, text="Enter the value of k:")
            k_label.pack()
            k_entry = tk.Entry(k_window)
            k_entry.pack()
            confirm_button = tk.Button(k_window, text="OK", command=on_k_submit)
            confirm_button.pack()
            k_window.wait_window()

            if k is not None:
                _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
                centers = np.uint8(centers)
                segmented_img = centers[labels.flatten()]
                segmented_img = segmented_img.reshape(i_img.shape)
                o_img = segmented_img.copy()

                # Display segmented image
                segmented_img = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(segmented_img)
                img.thumbnail((600, 400))
                img_tk = ImageTk.PhotoImage(img)
                canvas_o.create_image(0, 0, anchor=tk.NW, image=img_tk)
                canvas_o.image = img_tk
    except NameError:
        # Show error message
        message = "Please select an image first!"
        tk.messagebox.showerror(title="Error", message=message)

def mean_shift():
    global i_img, o_img
    try:
        if i_img is not None:
            spatial_radius = 20
            color_radius = 60
            max_pyramid_level = 1
            # Apply mean shift
            shifted = cv2.pyrMeanShiftFiltering(i_img, spatial_radius, color_radius, max_pyramid_level)
            o_img = shifted.copy()
            # Display segmented image
            shifted = cv2.cvtColor(shifted, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(shifted)
            img.thumbnail((600, 400))
            img_tk = ImageTk.PhotoImage(img)
            canvas_o.create_image(0, 0, anchor=tk.NW, image=img_tk)
            canvas_o.image = img_tk
    except NameError:
        # Show error message
        message = "Please select an image first!"
        tk.messagebox.showerror(title="Error", message=message)

def skeletonize():
    global i_img, o_img
    img = i_img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    ret, img = cv2.threshold(img, 135, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    img = Image.fromarray(skel)
    img.thumbnail((600, 400))
    img_tk = ImageTk.PhotoImage(img)
    canvas_o.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas_o.image = img_tk

    return skel

def extractMSERs():
    global i_img, o_img
    img = cv2.blur(i_img, (3, 3))
    # threshold the image
    ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    # extract MSERs
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(thresh)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    dst = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for contour in hulls:
        cv2.drawContours(dst, [contour], -1, (255, 255, 255), 1)
    img = Image.fromarray(dst)
    img.thumbnail((600, 400))
    img_tk = ImageTk.PhotoImage(img)
    canvas_o.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas_o.image = img_tk
    return dst

def Laplacian():
    global i_img, o_img
    ddepth = cv2.CV_16S
    kernel_size = 3
    src = cv2.GaussianBlur(i_img, (3, 3), 0)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst = cv2.Laplacian(src_gray, ddepth, ksize=kernel_size)
    abs_dst = cv2.convertScaleAbs(dst)
    img = Image.fromarray(abs_dst)
    img.thumbnail((600, 400))
    img_tk = ImageTk.PhotoImage(img)
    canvas_o.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas_o.image = img_tk
    return abs_dst

def extract_boudary():
    global i_img, o_img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erosion = cv2.erode(i_img, kernel, 1)
    boundary = i_img - erosion
    img = Image.fromarray(boundary)
    img.thumbnail((600, 400))
    img_tk = ImageTk.PhotoImage(img)
    canvas_o.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas_o.image = img_tk
    return boundary

# Create buttons
button1 = tk.Button(window, text="Select Image", command=select_image)
button2 = tk.Button(window, text="Save Image", command=save_image)
button3 = tk.Button(window, text="Exit", command=exit)
button4 = tk.Button(window, text="Show Histogram", command=show_histogram)
button5 = tk.Button(window, text="Gaussian Filter", command=gaussian_filter)
button6 = tk.Button(window, text="Equalize Hist", command=quantize_image)
button7 = tk.Button(window, text="Morphological Operation", command=morphological_operation)
button8 = tk.Button(window, text="K-Means++", command=k_means_pp)
button9 = tk.Button(window, text="Mean Shift", command=mean_shift)
button10 = tk.Button(window, text="Skeletonize", command=skeletonize)
button11 = tk.Button(window, text="Extract MSERs", command=extractMSERs)
button12 = tk.Button(window, text="Laplacian", command=Laplacian)
button13 = tk.Button(window, text="Extract Boundary", command=extract_boudary)

# Grid layout for widgets
label.grid(row=0, column=0, columnspan=2)
canvas_i.grid(row=1, column=0, padx=10, pady=10)
canvas_o.grid(row=1, column=1, padx=10, pady=10)
button1.grid(row=2, column=0, padx=10, pady=10)
button2.grid(row=2, column=1, padx=10, pady=10)
button3.grid(row=3, column=0, padx=10, pady=10)
button4.grid(row=3, column=1, padx=10, pady=10)
button5.grid(row=4, column=0, padx=10, pady=10)
button6.grid(row=4, column=1, padx=10, pady=10)
button7.grid(row=5, column=0, padx=10, pady=10)
button8.grid(row=5, column=1, padx=10, pady=10)
button9.grid(row=6, column=0, padx=10, pady=10)
button10.grid(row=6, column=1, padx=10, pady=10)
button11.grid(row=7, column=0, padx=10, pady=10)
button12.grid(row=7, column=1, padx=10, pady=10)
button13.grid(row=8, column=0, padx=10, pady=10)

window.mainloop()