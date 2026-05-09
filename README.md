# AI-Football-Detection

AI-Football-Detection is a computer vision pipeline that detects soccer players, classifies teams, tracks objects, and projects player positions onto a tactical radar map using YOLO, SigLIP-based team classification, ByteTrack, keypoint detection, and homography transformation.

## Overview

The system takes a soccer match video as input and generates a tactical pitch-view output showing detected player, referee, goalkeeper, and ball positions in real time.

## Pipeline

1. Load an input soccer match video in `.mp4` format.
2. Run a trained YOLO object detection model on each frame.
3. Detect and separate objects into:
   - Ball
   - Goalkeepers
   - Players
   - Referees
4. Crop detected player bounding boxes from sampled frames.
5. Train a team classifier on player crops to separate players into two teams.
6. Use ByteTrack to maintain object tracking across frames.
7. Assign goalkeeper team IDs based on proximity to each team's centroid.
8. Detect pitch keypoints from each frame.
9. Apply homography transformation to map field positions into a 2D tactical pitch view.
10. Render the output video with team-colored player locations, referee positions, and ball location.

## Core Features

- Player, goalkeeper, referee, and ball detection using YOLO
- Player crop extraction for team classification
- Team classification using visual embeddings and clustering
- Object tracking with ByteTrack
- Goalkeeper team assignment using player position centroids
- Pitch keypoint detection for field mapping
- Homography-based coordinate transformation
- Tactical radar visualization output

## Tech Stack

- Python
- YOLO (v8 & 26)
- Supervision
- ByteTrack
- NumPy
- OpenCV
- PyTorch
- SigLIP
- UMAP
- K-Means
- Homography Transformation

## Object Dectection Model Metrics
![Object Detection Metrics](/assets/object_detection_model_metrics.png)

## Object Dectection Model Example
![Object Detection Example with object id](/assets/object_detection_model_example_1.jpg)
![Object Detection Metrics with confidence score](/assets/object_detection_model_example_2.jpg)

## Crop Image for Team Classification
![Crop Image](/assets/crop_image.png)

## K-Means Result
![K-means Crop Image Result](/assets/k_means_results.png)
![K-Means Plot Result](/assets/football_clustering.ipynb%20-%20Colab%20-%20Google%20Chrome%205_8_2026%2010_33_44%20PM.png)



