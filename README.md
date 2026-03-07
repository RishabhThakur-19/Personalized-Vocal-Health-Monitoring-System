# Personalized Vocal Health Monitoring System

<div align="center">

![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Android](https://img.shields.io/badge/Platform-Mobile-informational)
![License](https://img.shields.io/badge/License-MIT-green)

**A Non-Invasive Health Monitoring Solution Using Voice Analysis**

</div>

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Objectives](#objectives)
3. [Proposed Methodology](#proposed-methodology)
4. [System Workflow](#system-workflow)
5. [Tools & Technologies](#tools--technologies)
6. [Implementation Timeline](#implementation-timeline)
7. [Expected Outcomes](#expected-outcomes)
8. [Team](#team)
9. [References](#references)

---

## 🎯 Overview

Early detection of illness using non-invasive methods is an emerging research field. Human voice carries important physiological information such as:
- Breathing irregularities
- Throat infections
- Fatigue
- Common cold
- Respiratory illnesses

### The Problem

Most voice-based health detection systems operate only on short, manually recorded samples and often fail in real-world environments where multiple people are speaking simultaneously.

### Our Solution

This project introduces a **personalized vocal health monitoring mobile application** designed for continuous, real-world operation. The system:
- ✅ Isolates the owner's voice in group conversations using advanced speaker recognition
- ✅ Enables passive monitoring without manual intervention
- ✅ Triggers health alerts only when abnormal vocal patterns are detected
- ✅ Significantly reduces false positives

---

## 🎯 Objectives

### Primary Objective

To develop a continuous, personalized voice-based health monitoring mobile application that detects potential illness using deep learning models integrated with real-time speaker verification.

### Specific Objectives

- Implement continuous microphone listening with optimized low power usage techniques
- Identify and isolate only the owner's voice in complex group environments
- Perform real-time speaker verification using high-fidelity embedding techniques
- Extract vocal health features (biomarkers) from detected speech segments
- Classify healthy vs. unhealthy voice states using deep learning (CNN/LSTM)
- Develop an intelligent alert system that filters temporary voice fluctuations

---

## 🔬 Proposed Methodology

### 3.1 Dataset Preparation

**Public Dataset:**
- Coswara dataset (respiratory samples from IISc Bangalore)

**Self-Recorded Dataset:**
- Owner samples in healthy and mild illness states recorded in diverse environments (indoor, outdoor, group conversations)

**Data Augmentation:**
- Background crowd noise, traffic noise, and reverberation to ensure real-world robustness

---

### 3.2 Continuous Voice Activity Detection (VAD)

A lightweight VAD system continuously monitors microphone input to:
- Extract only speech segments
- Ignore silence and non-human background noise
- Minimize processing overhead

---

### 3.3 Speaker Verification

**Setup Phase:**
- Owner records multiple samples to generate and store unique embeddings

**Monitoring Phase:**
- Incoming speech embeddings are compared with stored embeddings using Cosine Similarity
- If similarity surpasses the threshold → voice verified as owner
- Otherwise → audio disregarded (privacy protection)

---

### 3.4 Feature Extraction (Numerical Characteristics)

Confirmed owner segments are converted into numerical biomarkers:

**MFCC (Mel-Frequency Cepstral Coefficients):**
- Captures the fundamental sound characteristics
- Monitoring: Tracks frequency variations

**Pitch:**
- Represents fundamental frequency changes
- Detects hoarseness and vocal stress

**Energy:**
- Measures loudness and intensity
- Indicates vocal weakness or breathing difficulties

---

### 3.5 Illness Classification (Deep Learning)

A Convolutional Neural Network (CNN) / Long Short-Term Memory (LSTM) based model identifies:

**Healthy State:**
- Baseline vocal characteristics
- Normal respiratory patterns

**Possibly Ill State:**
- Hoarseness
- Reduced pitch range
- Irregular phrasing
- Anomalies in speech patterns

---

### 3.6 Alert Decision Logic

To ensure high reliability and minimize false positives:

- Requires multiple abnormal predictions before triggering an alert
- Evaluates consistent detection and short-term history
- Implements Pattern Persistence Logic

---

### 3.7 Android Mobile Application Development

Develop an Android application that:
- Runs continuous background voice monitoring
- Registers and stores the owner's voice embedding during setup
- Filters group conversations using speaker verification
- Sends notifications when illness patterns are detected
- Ensures secure processing of sensitive audio data

---

## 🔄 System Workflow

```
Microphone (Continuous Listening)
           ↓
Voice Activity Detection (VAD)
           ↓
Speaker Verification (Owner Detection)
           ↓
Feature Extraction (MFCC, Pitch, Energy)
           ↓
Illness Classification (Deep Learning Model)
           ↓
Pattern Persistence Logic
           ↓
Health Alert Notification
```

---

## 🛠️ Tools & Technologies

| Category | Technology / Tool |
|----------|-------------------|
| **Programming Language** | Python, React Native |
| **Deep Learning Framework** | PyTorch / TensorFlow |
| **Audio Processing** | Librosa (Python's library) |
| **Speaker Recognition** | Deep Speech Embeddings |
| **Backend Engine** | Flask / Node.js |
| **Mobile Application** | Android/React Native |
| **Dataset** | Coswara (IISc Bangalore), Recorded Voices |

---


## 🎁 Expected Outcomes

- ✨ **Precision Isolation:** Robust detection of the owner's voice even in noisy or group settings
- 🎧 **Passive Surveillance:** A system that works without requiring manual voice recording
- 🚨 **Early Warning:** Detection of respiratory issues before they become severe
- 🛡️ **High Reliability:** Low false-alarm rate due to multi-stage verification and persistence logic

---


## 📚 References

1. Sharma et al., *Coswara — A Database of Breathing, Cough and Voice Sounds for COVID-19 Diagnosis*, INTERSPEECH, 2020.

2. Brown et al., *Exploring Automatic Diagnosis of COVID-19 from Crowdsourced Respiratory Sound Data*, 2020.

3. FAIR model for respiratory disease detection, *Sensors Journal*, 2024.

---

<div align="center">

**Last Updated:** February 2026

Made with ❤️ by the Vocal Health Monitoring Team

</div>
