# Semantic-Features
Semantic Features for Localisation

# Tasks new:
Code clean up:
- All utility functions into utils.py (lidar to depth, disparity to depth, depth + mask -> landmark)
- Class for Mask R-CNN inference

Run KITTI dataset and visualize

# Tasks old:
Only use static classes for coco <br>
Find datasets with static features for training (all) <br>
representation? point based? <br>
Diff.GPS Data into map representation (Johannes) <br>
Geometric localization tool/representation

# Tasks old (deadline: monday 30.03.):
Johannes: Improve mapping - Heading angle and object classes <br>
Julius: Check if easier way to detect signs etc. and train detector on cityscapes <br>
Felix: Kitti dataset(for better GPS groundtruth) and helps Juuu and thinks about feature mapping <br>
Berk: Improve code and check if return values are true

# Object detector:
static classes mapillary vistas: class_names = ['Bench', 'Billboard', 'Catch Basin', 'CCTV Camera', 'Fire Hydrant', 'Junction Box', 'Mailbox',
               'Manhole', 'Phone Booth', 'Street Light', 'Pole', 'Traffic Sign Frame', 'Utility Pole',
               'Traffic Light', 'Traffic Sign (Back)', 'Traffic Sign (Front)', 'Trash Can']
mAP@IoU=0.5: 
class name: Bench: mAP: nan

class name: Billboard: mAP: 0.13378898509652173

class name: Catch Basin: mAP: 0.08723088352212927
class name: CCTV Camera: mAP: 0.009009009009009009
class name: Fire Hydrant: mAP: 0.05945945945945946
class name: Junction Box: mAP: 0.018995633194279982
class name: Mailbox: mAP: 0.0
class name: Manhole: mAP: 0.1898932374137172
class name: Phone Booth: mAP: 0.0
class name: Street Light: mAP: 0.16076993966647124
class name: Pole: mAP: 0.07435044233169
class name: Traffic Sign Frame: mAP: 0.0
class name: Utility Pole: mAP: 0.10728754850566499
class name: Traffic Light: mAP: 0.22378939301032894
class name: Traffic Sign (Back): mAP: 0.06839771660148428
class name: Traffic Sign (Front): mAP: 0.24776525653562692
class name: Trash Can: mAP: 0.06402141058740836
static classes : static_classes = ['other', 'traffic light', 'fire hydrant', 
                  'stop sign', 'parking meter', 'bench']

