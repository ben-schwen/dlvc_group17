# Evaluation of Depth Estimation 

## Installation

Use:
```
pip3 --install -r requirements.txt
```
to install everything.

Then download the AdaBins_nyu.pt weights from [here](https://drive.google.com/drive/folders/1nYyaQXOBjNdUJDsmJpcRpu6oE55aQoLA).

Next, the dataset can be download [here](https://drive.google.com/drive/folders/1sGAz7x7K2aPlONs6MhRLFI5gHPVVvdoM?usp=sharing), unzipped and placed at a known location.

Finally the camera has to be calibrated, with:

```
cd src
python3 calibrate.py --path path/to/unpacked/calibrated.zip
```

## Run aruco tests

```
python3 adabins.py --path ...
```

```
python3 aruco.py --source ...
```

## Run depth estimation

```
python3 depth.py --path ...
```

## Create Video
After generating all depth images, you can create a video with
```
python3 video.py
```

## Results
### Depth Comparison:
[![Depth Comparison](https://img.youtube.com/vi/x7cvzDrKnns/0.jpg)](https://www.youtube.com/watch?v=x7cvzDrKnns)

### Adabins vs ArUco1

[![Adabins vs ArUco1](https://img.youtube.com/vi/XK5LG37ycaw/0.jpg)](https://www.youtube.com/watch?v=XK5LG37ycaw)


### Adabins vs ArUco3

[![Adabins vs ArUco3](https://img.youtube.com/vi/09Sevrv2lBM/0.jpg)](https://www.youtube.com/watch?v=09Sevrv2lBM)


### Adabins vs ArUco4

[![Adabins vs ArUco2](https://img.youtube.com/vi/UrvkL1H2eDY/0.jpg)](https://www.youtube.com/watch?v=UrvkL1H2eDY)