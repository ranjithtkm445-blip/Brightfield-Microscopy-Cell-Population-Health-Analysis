

# Cell Health Analysis using Microscopy Images

**Live App:** [https://huggingface.co/spaces/Ranjith445/brightfield-cell-analysis](https://huggingface.co/spaces/Ranjith445/brightfield-cell-analysis)

---

## What is this project about?

Scientists often study cells using a microscope to understand their health.

One common method is **brightfield microscopy**, which does not require special dyes or equipment.

But analyzing these images manually is difficult because:

* Cells are not clearly separated
* Boundaries are hard to see
* Counting cells takes time
* Understanding cell health needs expertise

This project builds an **AI system that automatically analyzes cell images and gives a simple health report**.

---

## What does this system do?

This system:

* Looks at microscope images of cells
* Finds and outlines each cell
* Measures properties like size and shape
* Decides whether cells are healthy or not
* Summarizes the overall condition of the cell population

---

## How does it work (simple explanation)

The system works in two main steps:

---

### Step 1: Finding the cells

* The AI scans the image
* Detects where each cell is located
* Draws boundaries around cells

---

### Step 2: Checking cell health

For each cell, it measures:

* Size
* Shape
* Roundness
* Density

Then it compares these values with known biological ranges to decide:

* Healthy
* Slightly abnormal
* Stressed or damaged

---

## What data was used?

* Dataset: BBBC006 (from Broad Institute)
* Type: Brightfield microscopy images
* Total images: 768

From these images:

* Over **34,000 cells** were analyzed

---

## What results does it give?

* Cell detection performance is good
* Health classification accuracy: **99.96%**

The system also reports:

* Percentage of healthy cells
* Percentage of abnormal cells
* Overall cell condition

---

## What makes this project useful?

### Saves time

* Automatically analyzes hundreds of images

### Gives structured results

* Converts images into clear numbers and categories

### Easy to understand

* Provides simple health summaries

---

## Features of the application

* Upload or view sample cell images
* See cells outlined clearly
* View cell measurements
* Get health classification
* See overall population summary

---

## Important Note

* This system uses **rule-based thresholds** for health classification
* Dataset has **no drug or experiment labels**

This means:

* It describes cell condition
* It does not explain causes

---

## Limitations

* Works on a single time point only
* Cannot track cell changes over time
* May not work well on different datasets
* Model trained for limited number of epochs

---

## Future Improvements

* Train on more data
* Improve cell detection accuracy
* Add time-based analysis
* Use real biological labels

---

## Disclaimer

This project is for educational and research purposes only.
It should not be used for medical or laboratory decisions.

---

## One-Line Summary

An AI system that analyzes microscope images to detect cells and estimate how healthy they are.

---

## Author

Built by M. Ranjith Kumar as a biomedical AI portfolio project.

---

