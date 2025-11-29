# Wine Quality Prediction - MLOps Project

An end-to-end Machine Learning Operations (MLOps) pipeline for predicting wine quality using physicochemical properties. This project demonstrates industry best practices for building, training, evaluating, and deploying machine learning models with complete experiment tracking and a web interface for predictions.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Pipeline Components](#pipeline-components)
- [Configuration](#configuration)
- [Model Details](#model-details)
- [Web Application](#web-application)
- [Experiment Tracking](#experiment-tracking)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a complete MLOps workflow for predicting wine quality based on physicochemical tests. The system uses an ElasticNet regression model to predict wine quality scores (0-10) based on 11 input features including acidity, sugar content, pH levels, and alcohol percentage.

**Key Highlights:**
- End-to-end automated ML pipeline from data ingestion to model deployment
- ElasticNet regression model with hyperparameter tuning (alpha=0.2, l1_ratio=0.1)
- MLflow and DagHub integration for comprehensive experiment tracking
- Flask web application for real-time predictions
- Modular, production-ready code architecture
- Complete data validation and transformation pipeline

## ✨ Features

- **Automated Data Pipeline**: Downloads and processes wine quality dataset from GitHub
- **Data Validation**: Schema validation ensuring data integrity with 12 features
- **Train-Test Split**: Automated 75-25 split for model training and evaluation
- **ElasticNet Model**: L1/L2 regularized regression for robust predictions
- **Experiment Tracking**: Full MLflow and DagHub integration with metrics logging (RMSE, MAE, R²)
- **Model Registry**: Automated model versioning and registration in MLflow
- **Flask Web App**: User-friendly interface for real-time wine quality predictions
- **Configuration Management**: YAML-based configuration for easy hyperparameter tuning
- **Logging System**: Comprehensive logging for debugging and monitoring
- **Modular Architecture**: Separate components for each pipeline stage
- **Type Safety**: Type annotations with runtime validation using `ensure` library

## 🏗️ Project Architecture

The project follows a modular architecture with five distinct pipeline stages:

```
Data Ingestion → Data Validation → Data Transformation → Model Training → Model Evaluation
```

**Pipeline Flow:**
1. **Data Ingestion**: Downloads wine quality dataset (ZIP format) and extracts it
2. **Data Validation**: Validates 12 columns against schema (11 features + 1 target)
3. **Data Transformation**: Performs train-test split (75-25)
4. **Model Training**: Trains ElasticNet model with configured hyperparameters
5. **Model Evaluation**: Evaluates model performance and logs to MLflow/DagHub

Each stage is independently configurable and can be executed separately or as part of the complete pipeline.

## 📊 Dataset

**Source**: Red Wine Quality Dataset  
**URL**: https://github.com/krishnaik06/datasets/raw/refs/heads/main/winequality-data.zip  
**Size**: 1,599 samples × 12 features

**Input Features** (11):
- `fixed acidity`: Tartaric acid concentration (g/dm³)
- `volatile acidity`: Acetic acid concentration (g/dm³)
- `citric acid`: Citric acid concentration (g/dm³)
- `residual sugar`: Remaining sugar after fermentation (g/dm³)
- `chlorides`: Salt concentration (g/dm³)
- `free sulfur dioxide`: Free SO₂ concentration (mg/dm³)
- `total sulfur dioxide`: Total SO₂ concentration (mg/dm³)
- `density`: Wine density (g/cm³)
- `pH`: Acidity level (0-14 scale)
- `sulphates`: Potassium sulphate concentration (g/dm³)
- `alcohol`: Alcohol percentage (% vol)

**Target Variable**: `quality` (score between 0-10)

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

Execute all pipeline stages sequentially:

```bash
python main.py
```

This will run all five stages:
1. Data Ingestion
2. Data Validation
3. Data Transformation
4. Model Training
5. Model Evaluation (with MLflow logging)

### Running the Web Application

Start the Flask web server for predictions:

```bash
python app.py
```

The application will be available at `http://localhost:8080`

**Available Endpoints:**
- `GET /` - Home page with input form
- `GET /train` - Trigger model training pipeline
- `POST /predict` - Submit wine features for quality prediction

### Running Individual Pipeline Stages

Execute specific stages independently:

```python
from src.datascience.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.datascience.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
from src.datascience.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.datascience.pipeline.model_trainer_pipeline import ModelTrainerTrainingPipeline
from src.datascience.pipeline.model_evaluation_pipeline import ModelEvaluationTrainingPipeline

# Example: Run only data ingestion
pipeline = DataIngestionTrainingPipeline()
pipeline.initiate_data_ingestion()
```

### Making Predictions Programmatically

```python
from src.datascience.pipeline.prediction_pipeline import PredictionPipeline
import numpy as np

# Create prediction pipeline
predictor = PredictionPipeline()

# Sample wine features [fixed_acidity, volatile_acidity, citric_acid, ...]
features = np.array([[7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]])

# Get prediction
quality_score = predictor.predict(features)
print(f"Predicted Wine Quality: {quality_score}")
```

## 📁 Project Structure

```
MLops-Project/
├── .github/
│   └── workflows/           # CI/CD workflows (placeholder)
├── config/
│   └── config.yaml          # Pipeline configuration (paths, URLs)
├── src/
│   └── datascience/
│       ├── __init__.py      # Logging configuration
│       ├── components/      # Core pipeline components
│       │   ├── data_ingestion.py
│       │   ├── data_validation.py
│       │   ├── data_transformation.py
│       │   ├── model_trainer.py
│       │   └── model_evaluation.py
│       ├── config/
│       │   └── configuration.py  # Configuration manager
│       ├── constants/
│       │   └── __init__.py  # File paths constants
│       ├── entity/
│       │   └── config_entity.py  # Data classes for configs
│       ├── pipeline/        # Pipeline orchestration
│       │   ├── data_ingestion_pipeline.py
│       │   ├── data_validation_pipeline.py
│       │   ├── data_transformation_pipeline.py
│       │   ├── model_trainer_pipeline.py
│       │   ├── model_evaluation_pipeline.py
│       │   └── prediction_pipeline.py
│       └── utils/
│           └── common.py    # Utility functions (YAML, JSON, logging)
├── research/                # Jupyter notebooks for experimentation
│   ├── 1_data_ingestion.ipynb
│   ├── 2_data_validation.ipynb
│   ├── 3_data_transformation.ipynb
│   ├── 4_model_trainer.ipynb
│   └── 5_model_evaluation.ipynb
├── templates/               # Flask HTML templates
│   ├── index.html          # Input form
│   └── results.html        # Prediction results
├── artifacts/               # Generated artifacts (data, models, metrics)
│   ├── data_ingestion/
│   ├── data_validation/
│   ├── data_transformation/
│   ├── model_trainer/
│   └── model_evaluation/
├── logs/                    # Application logs
├── app.py                   # Flask web application
├── main.py                  # Main pipeline executor
├── params.yaml              # Model hyperparameters
├── schema.yaml              # Data schema definitions
├── requirements.txt         # Python dependencies
├── template.py              # Project structure generator
├── Dockerfile               # Docker configuration (placeholder)
└── README.md                # Project documentation
```── data_validation.py
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
