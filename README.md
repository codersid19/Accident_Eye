# Accident Eye - Real-time Road Accident Detection

Accident Eye is a Python-based application that detects road accidents from real-time CCTV footage and alerts the police with date, time, and location information. It utilizes a subset of the UCF Crime dataset and employs a deep learning model based on the 152-layer ResNet architecture implemented with PyTorch.

## Features

- Real-time road accident detection from CCTV footage.
- Alerts with timestamp and location information.
- Utilizes a deep learning model based on ResNet-152.
- Python and PyTorch are used for implementation.

## Dataset

The project uses a subset of the UCF Crime dataset, which contains various crime-related video clips. For road accident detection, a specific subset of this dataset is used.

## Requirements

- Python 3.x
- PyTorch
- OpenCV
- Other necessary Python libraries (see requirements.txt)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/YourUsername/Accident-Eye.git
   cd Accident-Eye
Usage
Prepare your own CCTV footage or use sample videos.

## Run the Accident Eye application:

bash
Copy code
run python mymodel.py
then run python testing.py
The application will process the video feed and detect accidents in real-time.

If an accident is detected, it will alert the police with timestamps and location details.


Acknowledgments
UCF Crime dataset
## Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these guidelines:

Fork the repository.
Create a new branch.
Make your changes.
Test your changes.
Submit a pull request.
