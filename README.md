# Adaptive Machine Learning Based IDS for Real-Time Zero Day Threat Detection and Response 

## Overview: 
This project proposes an Adaptive Machine Learning-Based Intrusion Detection System (IDS) designed to operate in real time and capable of detecting and responding to both known and unknown threats. The system leverages online learning algorithms, which update continuously as new data arrives, and combines them with chunk-based processing, making the architecture scalable and lightweight. Furthermore, by integrating real-time detection with feedback-based adaptation, the proposed IDS not only detects novel anomalies but also evolves its detection strategy dynamically.

## Key Features:

**Real-Time Threat Detection**: Continuous monitoring of network traffic to detect intrusions and anomalies in real-time.
Adaptive Learning: Uses machine learning models to automatically update and evolve its detection capabilities based on new data and attack patterns.

**Automated Response System**: Automatically initiates defensive actions such as isolating affected nodes or alerting system administrators upon detecting an attack.

**Scalability**: Built using a microservices architecture with Docker and Kubernetes, allowing for efficient scaling on cloud platforms like GCP.

**Custom Load Testing**: Includes tools for load and stress testing to measure how well the system performs under various traffic conditions.

**Cloud-Ready**: Configured for deployment on Google Cloud Platform (GCP) with managed Kubernetes clusters, supporting distributed workloads.