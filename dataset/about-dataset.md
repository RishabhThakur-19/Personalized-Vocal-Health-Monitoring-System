#### ***Coswara Dataset***





Coswara is a crowdsourced audio dataset created by researchers at the Indian Institute of Science (IISc) to help detect COVID-19 using sound.



&#x20;

#### ***The idea:***



Your voice, breathing, and cough patterns change when you have respiratory illnesses → deep learning models can learn these patterns.



What Data Does It Contain?



Each participant records multiple types of sounds:

#### 

#### ***Where Has Coswara Been Used?***



**Medical AI Research (Primary Use)**



Coswara has been widely used in healthcare-focused machine learning research, especially for:



Detecting COVID-19 from cough sounds

Analyzing breathing abnormalities

Identifying voice changes due to infection



Many research papers used Coswara to build non-invasive screening tools (no test kits required).

#### 

#### ***Audio Types***



Cough sounds (shallow \& heavy)

Breathing sounds (fast \& slow)

Speech recordings (reading digits / sentences)

Sustained vowel sounds (like "aaa")

Metadata Included

Age

Gender

COVID status (positive/negative/recovered)

Symptoms (fever, cough, etc.)

Location (coarse-level)





##### ***Folder Structure (Actual Dataset Style)***



coswara/

│

├── 2020-04-01/

│   ├── ID\_001/

│   │   ├── cough-heavy.wav

│   │   ├── breath-fast.wav

│   │   ├── speech.wav

│   │   └── metadata.json

│   │

│   └── ID\_002/

│       └── ...

│

├── 2020-04-02/

└── ...



&#x20;Organized by date → user → audio files



&#x20;Why It’s Useful for Deep Learning



This dataset is widely used for:



#### ***Tasks***



COVID detection (binary classification)

Multi-class classification (healthy vs symptomatic vs positive)

Audio-based biomarker extraction

Respiratory disease analysis





#### ***Strengths*** 



Large real-world dataset

Multi-modal audio (not just cough)

Useful for healthcare AI research

Open and actively used



#### ***Limitations*** 



Noisy recordings (crowdsourced)

Class imbalance (fewer positive cases)

Self-reported labels → may not be 100% accurate





