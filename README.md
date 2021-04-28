# Sign-language-recognition
Sequence-sign-language-recognition- with mediapipe Hands standalone

Installation (Building MediaPipe Python Package)

1.git clone mediapipe (https://github.com/google/mediapipe)

2.https://google.github.io/mediapipe/getting_started/install.html

3.https://google.github.io/mediapipe/getting_started/python.html

Step used

1.collect video input with path mediapipe/inputdata/class/*.mp4 ***only video.mp4 with no sound***

2.build.py --convert inputdata to filelandmark.txt in outputdata path

3.train.py --training absolute of data in outputdata path

4.predict_modify.py --predict sign language on real-time webcam from training model
