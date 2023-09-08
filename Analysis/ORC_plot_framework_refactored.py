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

def refactored_extract_graphs_from_pdf(pdf_path, pdf_name, pages):
    """
    Extract graphs from specified pages of a given PDF and save them as images.
    """
    # Convert PDF pages to images
    images = convert_from_path(pdf_path, dpi=300, first_page=min(pages), last_page=max(pages))

    # Base directory for saving images
    base_direc_path = "/mnt/data/Data/Pre_processing/images"
    
    all_extracted_images = []

    for page_num, img in zip(pages, images):
        # Convert PIL Image to NumPy array for processing
        img_np = np.array(img)

        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

        # Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Thresholding
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Finding contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Directory for saving extracted images for this PDF
        direc_path = os.path.join(base_direc_path, pdf_name)
        os.makedirs(direc_path, exist_ok=True)

        # Process each contour to extract potential graphs
        for index, contour in enumerate(contours):
            # If the contour area is above a threshold (e.g., to avoid tiny artifacts)
            if cv2.contourArea(contour) > 5000:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                extracted_img = img_np[y:y+h, x:x+w]

                # Save the extracted image
                image_path = os.path.join(direc_path, f"Page_{page_num}_Graph_{index}.png")
                cv2.imwrite(image_path, cv2.cvtColor(extracted_img, cv2.COLOR_RGB2BGR))
                all_extracted_images.append(image_path)

    return all_extracted_images


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

def save_data_points_to_csv(data_points, csv_filename, pdf_ohne_extension, method):
    """Save extracted data points to a CSV file."""
    directory = f"../Data/{method}" if method in ["Experimental", "Model"] else "../Data/Unknown"
    os.makedirs(directory, exist_ok=True)
    
    prefix = pdf_ohne_extension.split("_", 1)[0]  # Get the shortened title as a prefix
    csv_filename_with_prefix = f"{prefix}_{csv_filename}"
    
    with open(os.path.join(directory, csv_filename_with_prefix), 'w', newline='') as csvfile:
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
        graph_image_paths = refactored_extract_graphs_from_pdf(pdf_path, pdf_name_without_extension, pages)
        extracted_graphs[pdf_name_without_extension] = graph_image_paths
        
        # Pause for manual image review
        input(f"Images for {pdf_name_without_extension} have been extracted. Review and delete unwanted images, then press Enter to continue.")
        
        # Extract data points from the saved graph images and save to CSV
        # for image_path in graph_image_paths:
        #     data_points = extract_data_points_from_image(image_path)
        #     csv_filename = os.path.basename(image_path).replace(".png", ".csv")
        #     save_data_points_to_csv(data_points, csv_filename, pdf_name_without_extension, method_dict[pdf_name_without_extension])
    
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



