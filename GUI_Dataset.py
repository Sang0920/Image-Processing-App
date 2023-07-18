import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

global in_folder
global out_folder
window = tk.Tk()
window.title("Image Processing")

label = tk.Label(window, text="Image Processing", font=("Arial Bold", 20))
label_1 = tk.Label(window, text="Input Folder", font=("Arial Bold", 10))
label_2 = tk.Label(window, text="Output Folder", font=("Arial Bold", 10))
# label for displaying the input folder path
label_3 = tk.Label(window, text="", font=("Arial Bold", 10))
# label for displaying the output folder path
label_4 = tk.Label(window, text="", font=("Arial Bold", 10))

def select_in_folder():
    global in_folder
    in_folder = filedialog.askdirectory()
    label_3.configure(text=in_folder)
    print(in_folder)

def select_out_folder():
    global out_folder
    out_folder = filedialog.askdirectory()
    label_4.configure(text=out_folder)
    print(out_folder)

def gaussian_filter():
    global in_folder
    global out_folder
    
    if in_folder and out_folder:
        file_list = os.listdir(in_folder)
        for file_name in file_list:
            if file_name.endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(in_folder, file_name)
                img = cv2.imread(file_path)
                processed_img = cv2.GaussianBlur(img, (5, 5), 0)
                save_path = os.path.join(out_folder, file_name)
                cv2.imwrite(save_path, processed_img)
        print("Image processing completed.")
    else:
        print("Please select input and output folders.")

def quantize_image():
    global in_folder
    global out_folder

    if in_folder and out_folder:
        file_list = os.listdir(in_folder)
        for file_name in file_list:
            if file_name.endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(in_folder, file_name)
                img = cv2.imread(file_path)
                img_arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_arr = img_arr.astype(np.float32)
                img_arr /= 255.0
                img_arr = np.power(img_arr, 0.5)
                img_arr *= 255.0
                img_arr = img_arr.astype(np.uint8)
                save_path = os.path.join(out_folder, file_name)
                cv2.imwrite(save_path, img_arr)
        print("Image processing completed.")

def morphological_operation():
    global in_folder, out_folder
    try:
        if in_folder is None:
            pass
    except NameError:
        # show error message
        message = "Please select an input folder first!"
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

    if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
            # show error message
        message = "Kernel size should be odd!"
        tk.messagebox.showerror(title="Error", message=message)
    else:
        if in_folder and out_folder:
            file_list = os.listdir(in_folder)
            for file_name in file_list:
                if file_name.endswith((".png", ".jpg", ".jpeg")):
                    file_path = os.path.join(in_folder, file_name)
                    i_img = cv2.imread(file_path)
                    o_img = None

                    kernel = cv2.getStructuringElement(kernel_type, kernel_size)
                    processed_img = cv2.morphologyEx(i_img, operation, kernel)
                    o_img = processed_img.copy()
                    
                    if o_img is not None:
                        save_path = os.path.join(out_folder, file_name)
                        cv2.imwrite(save_path, o_img)
            print("Image processing completed.")
        else:
            print("Please select input and output folders.")

def kmeans_pp():
    global in_folder, out_folder
    try:
        if in_folder is None:
            pass
    except NameError:
        # show error message
        message = "Please select an input folder first!"
        tk.messagebox.showerror(title="Error", message=message)
    cluster_num = 0
    # show popup window to get number of clusters
    def on_cluster_num_submit():
        nonlocal cluster_num
        cluster_num = int(cluster_num_entry.get())
        cluster_num_window.destroy()

    cluster_num_window = tk.Toplevel(window)
    cluster_num_window.title("Number of Clusters")
    cluster_num_label = tk.Label(cluster_num_window, text="Enter the number of clusters:")
    cluster_num_label.pack()
    cluster_num_entry = tk.Entry(cluster_num_window)
    cluster_num_entry.pack()
    confirm_button = tk.Button(cluster_num_window, text="OK", command=on_cluster_num_submit)
    confirm_button.pack()
    cluster_num_window.wait_window()

    if in_folder and out_folder:
        file_list = os.listdir(in_folder)
        for file_name in file_list:
            if file_name.endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(in_folder, file_name)
                img = cv2.imread(file_path)
                Z = img.reshape(-1, 1)
                Z = np.float32(Z)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                _, labels, centers = cv2.kmeans(Z, cluster_num, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
                centers = np.uint8(centers)
                segmented_img = centers[labels.flatten()]
                segmented_img = segmented_img.reshape(img.shape)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                save_path = os.path.join(out_folder, file_name)
                cv2.imwrite(save_path, segmented_img)
        print("Image processing completed.")

def mean_shift():
    global in_folder, out_folder
    try:
        if in_folder is None:
            pass
    except NameError:
        # show error message
        message = "Please select an input folder first!"
        tk.messagebox.showerror(title="Error", message=message)
    
    spatial_radius, color_radius, max_pyramid_level = None, None, None

    # show popup window to get spatial radius
    def on_spatial_radius_submit():
        nonlocal spatial_radius
        spatial_radius = int(spatial_radius_entry.get())
        spatial_radius_window.destroy()

    spatial_radius_window = tk.Toplevel(window)
    spatial_radius_window.title("Spatial Radius")
    spatial_radius_label = tk.Label(spatial_radius_window, text="Enter the spatial radius:")
    spatial_radius_label.pack()
    spatial_radius_entry = tk.Entry(spatial_radius_window)
    spatial_radius_entry.pack()
    confirm_button = tk.Button(spatial_radius_window, text="OK", command=on_spatial_radius_submit)
    confirm_button.pack()
    spatial_radius_window.wait_window()

    # show popup window to get color radius
    def on_color_radius_submit():
        nonlocal color_radius
        color_radius = int(color_radius_entry.get())
        color_radius_window.destroy()

    color_radius_window = tk.Toplevel(window)
    color_radius_window.title("Color Radius")
    color_radius_label = tk.Label(color_radius_window, text="Enter the color radius:")
    color_radius_label.pack()
    color_radius_entry = tk.Entry(color_radius_window)
    color_radius_entry.pack()
    confirm_button = tk.Button(color_radius_window, text="OK", command=on_color_radius_submit)
    confirm_button.pack()
    color_radius_window.wait_window()

    # show popup window to get max pyramid level
    def on_max_pyramid_level_submit():
        nonlocal max_pyramid_level
        max_pyramid_level = int(max_pyramid_level_entry.get())
        max_pyramid_level_window.destroy()

    max_pyramid_level_window = tk.Toplevel(window)
    max_pyramid_level_window.title("Max Pyramid Level")
    max_pyramid_level_label = tk.Label(max_pyramid_level_window, text="Enter the max pyramid level:")
    max_pyramid_level_label.pack()
    max_pyramid_level_entry = tk.Entry(max_pyramid_level_window)
    max_pyramid_level_entry.pack()
    confirm_button = tk.Button(max_pyramid_level_window, text="OK", command=on_max_pyramid_level_submit)
    confirm_button.pack()
    max_pyramid_level_window.wait_window()

    if in_folder and out_folder:
        file_list = os.listdir(in_folder)
        for file_name in file_list:
            if file_name.endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(in_folder, file_name)
                img = cv2.imread(file_path)
                shifted = cv2.pyrMeanShiftFiltering(img, spatial_radius, color_radius, max_pyramid_level)
                save_path = os.path.join(out_folder, file_name)
                cv2.imwrite(save_path, shifted)
        print("Image processing completed.")
    else:
        print("Please select input and output folders.")

def skeletonize():
    global in_folder, out_folder
    try:
        if in_folder is None:
            pass
    except NameError:
        # show error message
        message = "Please select an input folder first!"
        tk.messagebox.showerror(title="Error", message=message)
    
    if in_folder and out_folder:
        file_list = os.listdir(in_folder)
        for file_name in file_list:
            if file_name.endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(in_folder, file_name)
                img = cv2.imread(file_path)
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
                save_path = os.path.join(out_folder, file_name)
                cv2.imwrite(save_path, skel)
        print("Image processing completed.")
    else:
        print("Please select input and output folders.")

def extractMSERs():
    global in_folder, out_folder
    try:
        if in_folder is None:
            pass
    except NameError:
        # show error message
        message = "Please select an input folder first!"
        tk.messagebox.showerror(title="Error", message=message)
    
    if in_folder and out_folder:
        file_list = os.listdir(in_folder)
        for file_name in file_list:
            if file_name.endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(in_folder, file_name)
                img = cv2.imread(file_path)
                img = cv2.blur(img, (3, 3))
                # threshold the image
                ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
                # extract MSERs
                mser = cv2.MSER_create()
                regions, _ = mser.detectRegions(thresh)
                hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
                dst = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
                for contour in hulls:
                    cv2.drawContours(dst, [contour], -1, (255, 255, 255), 1)
                save_path = os.path.join(out_folder, file_name)
                cv2.imwrite(save_path, dst)
        print("Image processing completed.")
    else:
        print("Please select input and output folders.")

def extract_boudary():
    global in_folder, out_folder
    try:
        if in_folder is None:
            pass
    except NameError:
        # show error message
        message = "Please select an input folder first!"
        tk.messagebox.showerror(title="Error", message=message)
    
    if in_folder and out_folder:
        file_list = os.listdir(in_folder)
        for file_name in file_list:
            if file_name.endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(in_folder, file_name)
                img = cv2.imread(file_path)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                erosion = cv2.erode(img, kernel, 1)
                boundary = img - erosion
                save_path = os.path.join(out_folder, file_name)
                cv2.imwrite(save_path, boundary)
        print("Image processing completed.")
    else:
        print("Please select input and output folders.")



button_1 = tk.Button(window, text="Select input folder", command=select_in_folder)
button_2 = tk.Button(window, text="Select output folder", command=select_out_folder)
button_3 = tk.Button(window, text="Gaussian Filter", command=gaussian_filter)
button_4 = tk.Button(window, text="Equalize Hist", command=quantize_image)
button_5 = tk.Button(window, text="Morphological Operation", command=morphological_operation)
button_6 = tk.Button(window, text="K-means++", command=kmeans_pp)
button_7 = tk.Button(window, text="Mean Shift", command=mean_shift)
button_8 = tk.Button(window, text="Skeletonize", command=skeletonize)
button_9 = tk.Button(window, text="Extract MSERs", command=extractMSERs)
button_10 = tk.Button(window, text="Extract Boundary", command=extract_boudary)

label.grid(column=0, row=0)
label_1.grid(column=0, row=1)
label_2.grid(column=0, row=2)
label_3.grid(column=1, row=1)
label_4.grid(column=1, row=2)
button_1.grid(column=1, row=1)
button_2.grid(column=1, row=2)
button_3.grid(column=0, row=3)
button_4.grid(column=1, row=3)
button_5.grid(column=2, row=3)
button_6.grid(column=3, row=3)
button_7.grid(column=4, row=3)
button_8.grid(column=5, row=3)
button_9.grid(column=6, row=3)
button_10.grid(column=7, row=3)

window.mainloop()
