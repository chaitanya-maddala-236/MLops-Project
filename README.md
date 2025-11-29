# MLOps Project

An end-to-end Machine Learning Operations (MLOps) pipeline demonstrating best practices for building, training, evaluating, and deploying machine learning models.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Pipeline Components](#pipeline-components)
- [Configuration](#configuration)
- [Model Tracking](#model-tracking)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a complete MLOps workflow that encompasses the entire machine learning lifecycle from data ingestion to model evaluation. The pipeline is designed to be modular, scalable, and reproducible, following industry best practices for machine learning operations.

## ✨ Features

- **End-to-End Pipeline**: Complete ML workflow from data ingestion to model evaluation
- **Data Validation**: Automated schema validation and data quality checks
- **Feature Engineering**: Comprehensive data transformation and preprocessing
- **Experiment Tracking**: Integration with MLflow and DagHub for experiment management
- **Modular Architecture**: Clean, maintainable code structure with separate components
- **Configuration Management**: Centralized configuration using YAML files
- **Reproducibility**: Consistent results through structured pipelines

## 🏗️ Project Architecture

The project follows a modular architecture with the following key stages:

```
Data Ingestion → Data Validation → Data Transformation → Model Training → Model Evaluation
```

Each stage is independently configurable and can be executed separately or as part of the complete pipeline.

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/chaitanya-maddala-236/MLops-Project.git
   cd MLops-Project
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up configuration files**
   - Update `config/config.yaml` with your project settings
   - Configure `config/schema.yaml` for data validation rules
   - Adjust hyperparameters in `config/params.yaml`

## 💻 Usage

### Running the Complete Pipeline

Execute the entire ML pipeline:

```bash
python main.py
```

### Running Individual Stages

You can also run individual pipeline stages:

```python
from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from src.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline

# Run specific stage
pipeline = DataIngestionTrainingPipeline()
pipeline.main()
```

## 📁 Project Structure

```
MLops-Project/
├── config/
│   ├── config.yaml          # Main configuration file
│   ├── schema.yaml          # Data schema definitions
│   └── params.yaml          # Model hyperparameters
├── src/
│   ├── components/          # Core pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   ├── config/              # Configuration manager
│   │   └── configuration.py
│   ├── entity/              # Data entities and configs
│   ├── pipeline/            # Pipeline stages
│   └── utils/               # Utility functions
├── artifacts/               # Generated artifacts
├── notebooks/               # Jupyter notebooks for exploration
├── research/                # Experimental code
├── main.py                  # Main pipeline executor
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## 🔧 Pipeline Components

### 1. Data Ingestion
- Fetches data from configured sources
- Handles data downloading and extraction
- Organizes raw data for processing

### 2. Data Validation
- Validates data against defined schema
- Checks data types and column names
- Generates validation status reports

### 3. Data Transformation
- Performs feature engineering
- Handles missing values and outliers
- Applies data preprocessing techniques
- Splits data into training and testing sets

### 4. Model Training
- Trains machine learning models with configured parameters
- Supports multiple algorithms
- Saves trained models for evaluation

### 5. Model Evaluation
- Evaluates model performance using various metrics
- Logs results to MLflow and DagHub
- Generates performance reports and visualizations

## ⚙️ Configuration

The project uses three main configuration files:

### config.yaml
Main configuration for pipeline components including:
- Data source locations
- Artifact directories
- Component-specific settings

### schema.yaml
Defines the expected data structure:
- Column names and types
- Target variable specifications
- Data validation rules

### params.yaml
Machine learning hyperparameters:
- Model-specific parameters
- Training configurations
- Evaluation metrics

## 📊 Model Tracking

This project integrates with **MLflow** and **DagHub** for comprehensive experiment tracking:

- **Experiment Logging**: All training runs are automatically logged
- **Metrics Tracking**: Performance metrics are recorded for each experiment
- **Model Registry**: Trained models are versioned and stored
- **Visualization**: Compare experiments and visualize model performance

### Setting up MLflow/DagHub

1. Create an account on [DagHub](https://dagshub.com/)
2. Set up your credentials
3. Update the tracking URI in your configuration
4. Run experiments and view results in the DagHub interface

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Chaitanya Maddala**

- GitHub: [@chaitanya-maddala-236](https://github.com/chaitanya-maddala-236)

## 🙏 Acknowledgments

- MLflow for experiment tracking capabilities
- DagHub for model versioning and collaboration
- The open-source community for various tools and libraries used in this project

---

For questions or support, please open an issue in the GitHub repository.
