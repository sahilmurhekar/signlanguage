# Hand-Sign Recognition Model Project

This project guides you through the process of collecting images, creating a dataset, training a hand-sign recognition model, and testing it. Follow the steps below to set up and run the project.

## Prerequisites
- Python 3.x installed on your system
- Required dependencies (see `requirements.txt` for details)
- A webcam or image files for collecting face data

## Installation
Clone this repository to your local machine:
```bash
git clone https://github.com/sahilmurhekar/signlanguage
cd signlanguage
```

## Streamlit App
Install the requirements and test the app on Streamlit:
```bash
pip install -r requirements.txt
streamlit run app.py
```
## Main Workflow
1. Run the following script to capture images of your face and create a dataset
```bash
python collect_images.py
```
2. Run the script below to preprocess the collected images and generate a data.pickle file
```bash
python create_dataset.py
```
3. Train the face recognition classifier using
```bash
python train_classifier.py
```
4. Run the inference script to test the trained model
```bash
python inference_classifier.py
```
## Notes
1. Ensure your webcam is connected (if applicable) when running collect_images.py.
2. The alternative Streamlit method (app.py) provides a user-friendly interface and may require additional setup depending on your environment.
