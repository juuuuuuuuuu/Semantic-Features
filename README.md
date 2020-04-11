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
static classes : static_classes = ['other', 'traffic light', 'fire hydrant', 
                  'stop sign', 'parking meter', 'bench']
