# SafeVision AR — Multi-Camera Computer Vision Safety Monitoring

A real-time **computer vision and augmented-reality safety monitoring prototype** combining multi-camera video analysis, YOLO-based detection/tracking, pose analysis, event reasoning and a Flutter AR client.

The project explores how visual AI can turn continuous camera streams into structured safety alerts that are easier for an operator to understand and act on.

## Recognition

🏆 **2nd Place — DIGILOG Process Optimization Challenge (May 2025)**

The **SIVAR / SafeVision project** earned second place in the DIGILOG Process Optimization Challenge held at ENET’Com Sfax from **23–27 May 2025**. The official certificate lists the challenge as organized by **ENET’Com (Tunisia), Leipzig University (Germany), and KNUST (Ghana)**, with DAAD/BMZ support.

This recognition is especially meaningful for the project because the challenge evaluated a complete applied solution rather than an isolated model: visual perception, event reasoning, real-time system integration and operator-facing delivery all had to work together.

[View the research portfolio and award section](https://rostom-ben-abdallah.github.io/#award)

## Demo

The video below shows the SafeVision prototype in operation and serves as a compact end-to-end demonstration of the project.

https://github.com/user-attachments/assets/1d871f73-b5b0-446e-bc65-86a07baaf2be

## Computer vision scope

- multi-camera video processing
- YOLO object detection and tracking
- pose-based fall detection
- fine-grained smoking-event reasoning using cigarette detection + face/lip geometry
- unattended-object / luggage alerts
- temporal event logic and track IDs
- real-time visualization and alert streaming
- AR overlays for operator-facing situational awareness

## High-level architecture

```text
Camera streams
     |
     v
YOLO detection / tracking
     |
     +------> object-risk reasoning
     |
     +------> pose estimation -> fall analysis
     |
     +------> face / lip geometry -> smoking-event reasoning
     |
     v
Structured safety events
     |
     v
WebSocket transport
     |
     v
Flutter / AR visualization client
```

## Why this project is relevant to my CV work

This project combines several themes that also appear in my later research and industrial work: **detection, tracking, temporal reasoning, pose analysis, multi-camera processing, event classification and real-time deployment**.

It is especially useful as a public example of how I connect computer-vision models to a complete application rather than treating inference as an isolated step.

## Technologies

`Python` · `OpenCV` · `Ultralytics YOLO` · `MediaPipe` · `ONNX Runtime` · `BoT-SORT` · `WebSockets` · `Flutter` · `AR`

## Repository layout

```text
├── codes/           # Python computer-vision and alert-streaming prototypes
├── models/          # model artifacts used by the prototype
└── safevision_ar/   # Flutter / AR client
```

## Portfolio note

This repository is an academic/team prototype and contains iterative experimental code. It should be read as an engineering demonstration of multi-camera visual perception and event reasoning rather than as a polished production release.

For my current research direction, see my [computer-vision portfolio](https://github.com/Rostom-Ben-Abdallah) and the [animal behaviour vision case study](https://github.com/Rostom-Ben-Abdallah/Rostom-Ben-Abdallah/blob/main/portfolio/mitacs-animal-behavior-vision.md).

## Contributors

Developed by **Rostom Ben Abdallah**, Alya Hamrouni, Mohamed Mtibaa and Rasem Bali.
