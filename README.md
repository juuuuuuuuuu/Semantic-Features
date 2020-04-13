# Semantic-Features
Semantic Features for Localisation

# Tasks (due until Thursday):
Goal: Run KITTI dataset and visualize<br>

Berk: Get coordinates of objects - put funktion in utils file <br>
Johannes: Output disparity map with higher resolution & (Helps Berk with dataset) <br>
Juuuuu: Class for Mask R-CNN inference & find form for accuracy <br>
Flixi: Visialization of output - Open3D <br>

All utility functions into utils.py (lidar to depth, disparity to depth, depth + mask -> landmark)



# Tasks old:
Only use static classes for coco <br>
Find datasets with static features for training (all) <br>
representation? point based? <br>
Diff.GPS Data into map representation (Johannes) <br>
Geometric localization tool/representation


# Object detector:
static classes mapillary vistas: class_names = ['Bench', 'Billboard', 'Catch Basin', 'CCTV Camera', 'Fire Hydrant', 'Junction Box', 'Mailbox',
               'Manhole', 'Phone Booth', 'Street Light', 'Pole', 'Traffic Sign Frame', 'Utility Pole',
               'Traffic Light', 'Traffic Sign (Back)', 'Traffic Sign (Front)', 'Trash Can']
mAP@IoU=0.5: 
class name: Bench: mAP: nan <br>
class name: Billboard: mAP: 0.13378898509652173 <br>
class name: Catch Basin: mAP: 0.08723088352212927 <br>
class name: CCTV Camera: mAP: 0.009009009009009009 <br>
class name: Fire Hydrant: mAP: 0.05945945945945946 <br>
class name: Junction Box: mAP: 0.018995633194279982 <br>
class name: Mailbox: mAP: 0.0 <br>
class name: Manhole: mAP: 0.1898932374137172 <br>
class name: Phone Booth: mAP: 0.0 <br>
class name: Street Light: mAP: 0.16076993966647124 <br>
class name: Pole: mAP: 0.07435044233169 <br>
class name: Traffic Sign Frame: mAP: 0.0 <br>
class name: Utility Pole: mAP: 0.10728754850566499 <br>
class name: Traffic Light: mAP: 0.22378939301032894 <br>
class name: Traffic Sign (Back): mAP: 0.06839771660148428 <br>
class name: Traffic Sign (Front): mAP: 0.24776525653562692 <br>
class name: Trash Can: mAP: 0.06402141058740836 <br>
static classes : static_classes = ['other', 'traffic light', 'fire hydrant', 
                  'stop sign', 'parking meter', 'bench']

