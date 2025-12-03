# NeuroAdapt-X: Stress-Resilient Neural Decoder for Space Operations 🧠🚀

Reliable BCI decoding under spaceflight stressors (high-G, microgravity, isolation) is a NASA-identified gap (Behavioral Health risks). This project demonstrates an adaptive decoder that recovers accuracy when baselines fail.

## Live Demo
1. **Hardware**: Stream from OpenBCI/Muse via LSL.
2. **Control**: Imagine left/right hand to steer a rover.
3. **Stress Test**: Toggle stressor injection → watch baseline fail → adaptation recover.

Video demo: [Soon!]

## Quick Start
```bash
git clone https://github.com/josephjilovec/neuroadapt-x.git
cd neuroadapt-x
pip install -r requirements.txt

## Offline training:
python src/models/train.py

## Live demo:
python src/demo/rover_control.py --live  # Or --replay for recorded data
