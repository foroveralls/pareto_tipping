# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 17:27:51 2023

@author: everall
"""
import pandas as pd
import os
import cv2
import numpy as np
import csv
from pdf2image import convert_from_path


folder_path = "../Data/Lit"
image_path_og = "../Data/Pre_processing/images/"

#%%
def adjusted_csv_to_dict_v6(csv_path):
    """Convert a CSV file with columns 'title', 'pages', and 'method' to a dictionary.
    Handles titles with commas and multiple page numbers."""
    page_dict = {}
    method_dict = {}  # Dictionary to store the method ("Experimental" or "Model") for each title
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            # Identify the boundary where the page numbers start
            # It's the first numeric item after the title
            title_boundary = next(i for i, item in enumerate(row) if item.isdigit())
            title = ",".join(row[:title_boundary])
            pages = list(map(int, row[title_boundary:-1]))  # Excluding the method column
            method = row[-1]  # The last column is the "method"
            page_dict[title] = pages
            method_dict[title] = method
    return page_dict, method_dict  # Return both dictionaries


def modified_generate_page_dict(folder_path):
    """Auto-generate a dictionary of PDF filenames and corresponding pages for extraction."""
    page_dict = {}
    
    # Auto-generate page_dict values based on PDF filenames in the folder
    for pdf_filename in os.listdir(folder_path):
        if pdf_filename.endswith(".pdf"):
            pdf_name_without_extension = os.path.splitext(pdf_filename)[0]
            page_dict[pdf_name_without_extension] = []  # Empty list to be filled manually
    
    return page_dict

def extract_graphs_from_pdf_v3(pdf_path, pdf_name, pages):
    """
    Extract graphs from specified pages of a PDF and save them as images.
    """
    saved_image_paths = []

    for page_num in pages:
        # Convert the specified page to image
        page_image = convert_from_path(pdf_path, dpi=300, first_page=page_num, last_page=page_num)[0]
        page_image = np.array(page_image)
        
        # Convert to grayscale if it's in color
        if len(page_image.shape) == 3:
            page_image = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(page_image, (5, 5), 0)
        
        # Thresholding
        _, binary_image = cv2.threshold(blurred_image, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the area of the largest rectangle
        largest_rectangle_area = max([cv2.contourArea(c) for c in contours])
        
        # Filter contours based on their similarity in size to the largest rectangle
        size_threshold = 0.7  # Keep rectangles that are at least 70% the size of the largest rectangle
        filtered_contours = [c for c in contours if cv2.contourArea(c) > size_threshold * largest_rectangle_area]
        
        for index, contour in enumerate(filtered_contours):
            x, y, w, h = cv2.boundingRect(contour)
            
            # Add buffer to the rectangle
            buffer_percent = 0.05
            buffer_w = int(w * buffer_percent)
            buffer_h = int(h * buffer_percent)
            x = max(0, x - buffer_w)
            y = max(0, y - buffer_h)
            w += 2 * buffer_w
            h += 2 * buffer_h
            
            # Crop the image
            cropped_image = page_image[y:y+h, x:x+w]
            
            # Save the cropped image with a unique filename
            image_name = os.path.basename(pdf_path).replace(".pdf", f"_page_{page_num}_graph_{index}.png")
            pdf_name = pdf_name.split(" ")[0]
            rest_path = f"{pdf_name}/{image_name}"
            direc_path = os.path.join(os.path.dirname(image_path_og), pdf_name)
            image_path = os.path.join(os.path.dirname(image_path_og), rest_path)
            print(image_path)
        
            try:
                os.mkdir(direc_path)
            except OSError:
                print ("Creation of the directory %s failed" % direc_path)
            else:
                print ("Successfully created the directory %s " % direc_path)
            cv2.imwrite(image_path, cropped_image)
            saved_image_paths.append(image_path)
    
    return saved_image_paths

# Testing the updated function with the provided PDF


def extract_data_points_from_image(image_path, x_axis_limits=None, y_axis_limits=None):
    """
    Extract data points from a plot in an image.
    
    Parameters:
    - image_path: Path to the image containing the plot.
    - x_axis_limits: Tuple containing the minimum and maximum values of the x-axis.
    - y_axis_limits: Tuple containing the minimum and maximum values of the y-axis.
    
    Returns:
    - List of (x, y) coordinates of the extracted data points.
    """
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Preprocess the image
    img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, img_binary = cv2.threshold(img_blurred, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Detect edges
    edges = cv2.Canny(img_binary, 50, 150)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=10, maxLineGap=5)
    
    # Extract endpoints of the detected lines as data points
    points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        points.append((x1, y1))
        points.append((x2, y2))
    
    # Sort the data points by their x-coordinates
    points = sorted(points, key=lambda x: x[0])
    
    # If axis limits are provided, normalize the data points
    if x_axis_limits and y_axis_limits:
        x_min, x_max = x_axis_limits
        y_min, y_max = y_axis_limits
        img_width, img_height = img.shape[1], img.shape[0]
        points = [((x / img_width) * (x_max - x_min) + x_min, 
                   (1 - (y / img_height)) * (y_max - y_min) + y_min) for x, y in points]
    
    return points

def save_data_points_to_csv(data_points, csv_filename, method):
    """Save extracted data points to a CSV file."""
    directory = f"../Data/{method}" if method in ["Experimental", "Model"] else "../Data/Unknown"
    os.makedirs(directory, exist_ok=True)
    
    with open(os.path.join(directory, csv_filename), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y"])  # Column headers
        for point in data_points:
            writer.writerow(point)


def process_pdfs(folder_path, page_dict):
    """Extract graphs from all PDFs in a folder based on the provided page_dict and save data points to CSV."""
    extracted_graphs = {}
    
    for pdf_name_without_extension, pages in page_dict.items():
        pdf_path = os.path.join(folder_path, pdf_name_without_extension + '.pdf')
        
        # Extract graphs from the PDF
        graph_image_paths = extract_graphs_from_pdf_v3(pdf_path, pdf_name_without_extension, pages)
        extracted_graphs[pdf_name_without_extension] = graph_image_paths
        
        # Extract data points from the saved graph images and save to CSV
        for image_path in graph_image_paths:
            data_points = extract_data_points_from_image(image_path)
            csv_filename = os.path.join(folder_path, pdf_name_without_extension + '_data_points.csv')
            save_data_points_to_csv(data_points, csv_filename)
    
    return extracted_graphs

#%%
#get csv of pdf's to give page numbers
# file_dic = modified_generate_page_dict(folder_path)
# csv_dic = pd.DataFrame.from_dict(file_dic, orient ="index").rename_axis("title").reset_index()
# csv_dic["pages"] = 0
# csv_dic["method"] = ""  # Adding the "method" column
# csv_dic.to_csv("../Data/Pre_processing/page_directory.csv", index = False)
#%%
#load csv
page_dict, method_dict = adjusted_csv_to_dict_v6("../Data/Pre_processing/page_directory.csv")
#
process_pdfs(folder_path, page_dict)

