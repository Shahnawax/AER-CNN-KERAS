## AER-CNN-KERAS
This project is a proof of the concept implementation of a Convolutioinal Neural Network (`CNN`) based implementation of Audio Event Recognition (`AER`) in KERAS. Keep in mind that it is the first version by the author and is more like an over night build.  As it is, it is not the best model available for this prupose. The work on the model is under progress and any refinements will be updated on the repository.

### Tools Required

Python 3.6 is used during development and following libraries are required to run the code provided in the notebook:
* `keras 2.x.`
* `numpy`
* `librosa`

### Database used

The **[ESC-50 dataset](https://github.com/karoldvl/ESC-50)** is a public labeled set of 2000 environmental recordings (50 classes, 40 clips per class, approximately 5 seconds per clip) suitable for environmental sound classification tasks.

See **[ESC: Dataset for Environmental Sound Classification - paper replication data](https://github.com/karoldvl/paper-2015-esc-dataset)** for the full paper with a more thorough analysis.

The available sound classes arranged alphabatically are given below: 
- **[1 - Airplane](https://github.com/karoldvl/ESC-50/tree/master/508%20-%20Airplane)**
- **[2 - Breathing](https://github.com/karoldvl/ESC-50/tree/master/304%20-%20Breathing)**
- **[3 - Brushing teeth](https://github.com/karoldvl/ESC-50/tree/master/308%20-%20Brushing%20teeth)**
- **[4 - Can opening](https://github.com/karoldvl/ESC-50/tree/master/405%20-%20Can%20opening)**
- **[5 - Cat](https://github.com/karoldvl/ESC-50/tree/master/106%20-%20Cat)**
- **[6 - Car horn](https://github.com/karoldvl/ESC-50/tree/master/504%20-%20Car%20horn)**
- **[7 - Chainsaw](https://github.com/karoldvl/ESC-50/tree/master/502%20-%20Chainsaw)**
- **[8 - Chirping birds](https://github.com/karoldvl/ESC-50/tree/master/205%20-%20Chirping%20birds)**
- **[9 - Church bells](https://github.com/karoldvl/ESC-50/tree/master/507%20-%20Church%20bells)**
- **[10 - Clapping](https://github.com/karoldvl/ESC-50/tree/master/303%20-%20Clapping)**
- **[11 - Clock alarm](https://github.com/karoldvl/ESC-50/tree/master/408%20-%20Clock%20alarm)**
- **[12 - Clock tick](https://github.com/karoldvl/ESC-50/tree/master/409%20-%20Clock%20tick)**
- **[13 - Coughing](https://github.com/karoldvl/ESC-50/tree/master/305%20-%20Coughing)**
- **[14 - Cow](https://github.com/karoldvl/ESC-50/tree/master/104%20-%20Cow)**
- **[15 - Crackling fire](https://github.com/karoldvl/ESC-50/tree/master/203%20-%20Crackling%20fire)**
- **[16 - Crickets](https://github.com/karoldvl/ESC-50/tree/master/204%20-%20Crickets)**
- **[17 - Crow](https://github.com/karoldvl/ESC-50/tree/master/110%20-%20Crow)**
- **[18 - Crying baby](https://github.com/karoldvl/ESC-50/tree/master/301%20-%20Crying%20baby)**
- **[19 - Dog](https://github.com/karoldvl/ESC-50/tree/master/101%20-%20Dog)**
- **[20 - Door knock](https://github.com/karoldvl/ESC-50/tree/master/401%20-%20Door%20knock)**
- **[21 - Door - wood creaks](https://github.com/karoldvl/ESC-50/tree/master/404%20-%20Door%20-%20wood%20creaks)**
- **[22 - Drinking - sipping](https://github.com/karoldvl/ESC-50/tree/master/310%20-%20Drinking%20-%20sipping)**
- **[23 - Engine](https://github.com/karoldvl/ESC-50/tree/master/505%20-%20Engine)**
- **[24 - Fireworks](https://github.com/karoldvl/ESC-50/tree/master/509%20-%20Fireworks)**
- **[25 - Footsteps](https://github.com/karoldvl/ESC-50/tree/master/306%20-%20Footsteps)**
- **[26 - Frog](https://github.com/karoldvl/ESC-50/tree/master/105%20-%20Frog)**
- **[27 - Glass breaking](https://github.com/karoldvl/ESC-50/tree/master/410%20-%20Glass%20breaking)**
- **[28 - Hand saw](https://github.com/karoldvl/ESC-50/tree/master/510%20-%20Hand%20saw)**
- **[29 - Helicopter](https://github.com/karoldvl/ESC-50/tree/master/501%20-%20Helicopter)**
- **[30 - Hen](https://github.com/karoldvl/ESC-50/tree/master/107%20-%20Hen)**
- **[31 - Insects (flying)](https://github.com/karoldvl/ESC-50/tree/master/108%20-%20Insects)**
- **[32 - Keyboard typing](https://github.com/karoldvl/ESC-50/tree/master/403%20-%20Keyboard%20typing)**
- **[33 - Laughing](https://github.com/karoldvl/ESC-50/tree/master/307%20-%20Laughing)**
- **[34 - Mouse click](https://github.com/karoldvl/ESC-50/tree/master/402%20-%20Mouse%20click)**
- **[35 - Pig](https://github.com/karoldvl/ESC-50/tree/master/103%20-%20Pig)**
- **[36 - Pouring water](https://github.com/karoldvl/ESC-50/tree/master/208%20-%20Pouring%20water)**
- **[37 - Rain](https://github.com/karoldvl/ESC-50/tree/master/201%20-%20Rain)**
- **[38 - Rooster](https://github.com/karoldvl/ESC-50/tree/master/102%20-%20Rooster)**
- **[39 - Sea waves](https://github.com/karoldvl/ESC-50/tree/master/202%20-%20Sea%20waves)**
- **[40 - Sheep](https://github.com/karoldvl/ESC-50/tree/master/109%20-%20Sheep)**
- **[41 - Siren](https://github.com/karoldvl/ESC-50/tree/master/503%20-%20Siren)**
- **[42 - Sneezing](https://github.com/karoldvl/ESC-50/tree/master/302%20-%20Sneezing)**
- **[43 - Snoring](https://github.com/karoldvl/ESC-50/tree/master/309%20-%20Snoring)**
- **[44 - Thunderstorm](https://github.com/karoldvl/ESC-50/tree/master/210%20-%20Thunderstorm)**
- **[45 - Toilet flush](https://github.com/karoldvl/ESC-50/tree/master/209%20-%20Toilet%20flush)**
- **[46 - Train](https://github.com/karoldvl/ESC-50/tree/master/506%20-%20Train)**
- **[47 - Vacuum cleaner](https://github.com/karoldvl/ESC-50/tree/master/407%20-%20Vacuum%20cleaner)**
- **[48 - Washing machine](https://github.com/karoldvl/ESC-50/tree/master/406%20-%20Washing%20machine)**
- **[49 - Water drops](https://github.com/karoldvl/ESC-50/tree/master/206%20-%20Water%20drops)**
- **[50 - Wind](https://github.com/karoldvl/ESC-50/tree/master/207%20-%20Wind)**.

## Experiments
### Preprocessing
First of all we renamed all the files in the classes to be numbers from `1` to `40`. Then all the files were read and we calculated the dBscale `Mel Spectrogram` with `n_mels = 128`. All the rest of the elements are left to be default in `librosa.features.melspectrogram`. All the files are of different length so in order to make sure that the preprocessed data has equal size for all the files we selected on `300` frames.
## Experimentation and data segmentation
We trained our model on all `50` classes. The total data is shuffeled in order to mix the classes and loose patterns. Then the data is divided into `2` subsets. `80%` for training and `20%` for testing. The training data is then further divided into `2` subsets with randomly selecting approaximately `80%` data for `trainig` and rest of the data for `validation`. So at the end we have `400` instances for testing `(8 files per class)`, approximately `1280` instance for training and `320` instance for validation. 
## Results
We tested the model for all the classes and got the overall average accuracy of `52%`. We foundout that our model performs relatively well on the repetitive sounds. Considering that we report the results of the `10` choosen classes namely, 

* Rain          `87.5%`
* Rooster       `87.5%`
* Door Knock    `100%`
* Siren         `100%`
* Clock Alarm   `87.5%`
* Clapping      `100%`
* Helicopter    `100%`
* Can Opening   `87.5%` 
* HandSaw       `87.5%`
* PouringWater  `87.5%` .

On these classes the accuracy of the model is `92.5%` on average.
