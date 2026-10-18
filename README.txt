GreenGuardian AI — FINAL STABLE CNN PROJECT

RUN APPLICATION
---------------
Double-click:
RUN_APP.bat

TRAIN CNN
---------
Use:
TRAIN_CNN_D_DRIVE.bat

Expected folders:
D:\Dataset\PlantVillage
D:\Dataset\Rose
D:\Dataset\Apple
D:\Dataset\Grape

Apple and Grape folders may start empty.
- AppleLeaf9 is downloaded automatically from the official GitHub repository.
- Grape GLVD is downloaded automatically from Zenodo.
- Rose must contain the official 7-class Rose dataset ZIP or extracted folder.
  If it is missing, the official Mendeley page opens.

FINAL CNN CLASS SCOPE
---------------------
Tomato — 10
Healthy Plant
Bacterial Spot
Early Blight
Late Blight
Leaf Mold
Septoria Leaf Spot
Spider Mites
Target Spot
Yellow Leaf Curl Virus
Mosaic Virus

Apple — 10
Healthy Plant
Apple Scab
Black Rot
Cedar Apple Rust
Alternaria Leaf Spot
Brown Spot
Frogeye Leaf Spot
Grey Spot
Mosaic
Powdery Mildew

Grape — 7
Healthy Plant
Black Rot
Esca (Black Measles)
Leaf Blight
Downy Mildew
Powdery Mildew
Bacterial Rot

Corn — 4
Healthy Plant
Common Rust
Gray Leaf Spot
Northern Leaf Blight

Rose — 7
Healthy Plant
Black Spot
Downy Mildew
Powdery Mildew
Rose Mosaic
Rose Rust
Insect Damage

TOTAL CNN CLASSES: 38

Separate rule-based smart-care conditions:
- Nutrient Deficiency
- Overwatering Stress
- General Plant Stress

DATA SOURCES
------------
PlantVillage:
https://github.com/spMohanty/PlantVillage-Dataset

AppleLeaf9:
https://github.com/JasonYangCode/AppleLeaf9
CC BY 4.0

Grapevine Leaf Variety & Disease Dataset (GLVD):
https://zenodo.org/records/18937397
DOI: 10.5281/zenodo.18937397

Rose:
https://data.mendeley.com/datasets/8jtfk9szbg/3
DOI: 10.17632/8jtfk9szbg.3
CC BY 4.0

TRAINING SAFETY / QUALITY
-------------------------
- Exact duplicate image files are removed before split.
- D-drive hardlinks are used where possible to reduce extra disk usage.
- Train / validation / test split is 70 / 15 / 15.
- Training refuses to start if any final target class is missing.
- Existing model files are reused only if their class list matches the final scope.
- MobileNetV2 preprocessing is inside the model graph.
- Low-confidence / low-margin CNN predictions are rejected as Uncertain.
- Evaluation outputs include accuracy, precision, recall, F1, classification report,
  confusion matrix and training curve.

IMPORTANT
---------
"All diseases" means all 38 disease/healthy classes defined and supported by the
verified datasets above. No finite image dataset contains every possible real-world
disease for these plant species. Unsupported patterns are rejected as Uncertain
instead of being forced into a false diagnosis.
