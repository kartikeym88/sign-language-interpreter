Sign-Language-Translator

Sign-Language-Translator is a project that translates sign language gestures into text using a Neural Network. It allows users to collect their own hand gesture data, train a model, and perform real-time predictions with grammar correction.

Key Features

Record custom hand gestures and create your own dataset.

Train an LSTM-based Neural Network to recognize gestures.

Predict hand gestures in real time using a camera.

Correct grammar in predicted sentences using language_tool_python.

Accurate hand tracking with MediaPipe Holistic.

How It Works

The project has three main steps:

Data Collection
Use data_collection.py
 to record gestures. You can set which signs to record, how many sequences, and where to save the dataset. MediaPipe Holistic tracks hand landmarks to create your dataset.



Model Training
The dataset is split into training (90%) and testing (10%). A simple Neural Network with LSTM and Dense layers is trained to recognize gestures.

Real-Time Predictions
The trained model processes live video frames, predicts gestures, and forms sentences. Press Enter to correct grammar or Spacebar to start over.

Prerequisites

Python 3.6+

Java 8.0+

language-tool-python (pip install language-tool-python)

mediapipe (pip install mediapipe)

tensorflow (pip install tensorflow)

How to Use

Collect data:

    python data_collection.py


Train the model:

    python train_model.py


Predict gestures in real time:

    python predict.py

Conclusion

Sign-Language-Translator makes it easy to convert sign language gestures into text. It helps bridge the communication gap for the deaf and hearing-impaired in a simple, intuitive way.
