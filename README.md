# CarEye
The simplest ADAS that detects and classifies road signs from the camera's video stream in real-time.
#### Description:
Cascade classifiers (two classifiers: for circular and for triangular and rectangular traffic signs) trained on LBP-features is used to detect road signs. The detected road signs are then filtered by the linear binary SVM-classifier (also two classifiers: for circular and for triangular and rectangular traffic signs) trained on HOG-features. Finally, the common multiclass SVM-classifier (one-vs-all scheme, with RBF kernel) learned on HOG-features predict classes of detected traffic signs.
#### Params of traffic sign detector:
Input image: size=1024x768<br>
Pyramid of images: scaleFactor=1.2<br>
NMS: minNeighbors=5<br>
Sliding window: minSize=24x24, maxSize=144x144
#### Test of traffic sign detector:
Cascade classifier for circular traffic signs:<br>
recall=76.8%, precision=78.5%, f1-score=77.6%<br>
Cascade classifier for triangular and rectangular traffic signs:<br>
recall=80.0%, precision=81.6%, f1-score=80.8%<br>
SVM-classifier for circular traffic signs:<br>
accuracy=98.6%<br>
SVM-classifier for triangular and rectangular traffic signs:<br>
accuracy=97.46%
#### Test of traffic sign classifier:
SVM-classifier:<br>
accuracy=96.5%
#### Perfomance test:
CPU: Intel Core i5-4570<br>
RAM: 8Gb<br>
Inference: 9-11 FPS (single thread)
