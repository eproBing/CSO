# Solar Radio Burst Detection and Analysis Pipeline

## Overview
This project provides an automated, end-to-end pipeline for detecting, segmenting, and analyzing solar radio bursts (e.g., Type II bursts) from dynamic spectra images. It bridges the gap between deep learning models and space weather forecasting systems.

The pipeline utilizes **YOLO11** for bounding box detection and **MobileSAM** for precise mask segmentation. It extracts critical physical parameters (such as frequency drift, bandwidth, and shock velocity using coronal density models) and outputs standardized L4 data products (JSON) and formatted images for operational space weather warning systems.

## Features
* **Object Detection:** Utilizes YOLO11 to identify radio burst regions in spectral images.
* **Instance Segmentation:** Uses MobileSAM to extract precise pixel-level masks of the bursts.
* **Physical Parameter Inversion:** Calculates physical metrics including start/end times, frequency ranges, and coronal shock velocities (via Newkirk/Saito models).
* **Data Standardization:** Automatically resizes images, applies strict naming conventions, and generates highly structured JSON files for downstream early-warning systems.

## Requirements
Ensure you have Python 3.8+ installed. Install the required dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
