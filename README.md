# CSE 132 Final Project: Automated Security Command Generation

## Overview
This project builds an automated pipeline that generates offensive security commands aligned with the **MITRE ATT&CK Framework**. The goal is to understand adversarial techniques and explore AI-assisted security testing tools through a **multi-agent system**.

## Project Goals
- Develop a **multi-agent pipeline** for generating security commands
- Leverage **Large Language Models (LLMs)** to automatically produce commands for specific ATT&CK techniques
- Validate command **correctness** and **relevance** to specified techniques
- Generate a dataset of **500–700 diverse, syntactically correct** security commands

## Technical Approach
The pipeline follows a **modular architecture** with specialized stages:

1. **Input Definition**  
   Specify a MITRE ATT&CK technique ID or description

2. **Retrieval (RAG)**  
   Extract relevant ATT&CK definitions and examples

3. **Generation**  
   Use LLM inference to create candidate commands

4. **Structuring**  
   Enforce predictable output schemas (JSON / Pydantic)

5. **Validation**  
   Verify syntactic validity and technique relevance

6. **Output**  
   Store results for analysis and benchmarking

## Tech Stack

**Infrastructure**
- Modal (cloud compute with GPU support)

**Models**
- Open-source LLMs (Phi, Llama 3.1)
- Model sizes ranging from **8B–70B parameters**

**Tools**
- Outlines (structured output enforcement)
- Retrieval-Augmented Generation (RAG) frameworks
- HuggingFace ecosystem

**Dataset**
- Bash commands aligned with MITRE ATT&CK techniques

## Deliverables
- **Public HuggingFace dataset**  
  500–700 samples in Parquet format

- **Comprehensive project report**  
  4+ pages covering methodology, challenges, and insights

- **Public Modal notebook**  
  Complete, reproducible implementation

## Notes
This project is intended for **educational and research purposes only**, focusing on understanding adversarial behavior and improving defensive security practices.

