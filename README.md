🚗 Car Price Prediction System
An end-to-end Machine Learning web application that predicts the resale price of a car based on its specifications, features, and history. The system trains a Random Forest Regressor model on historical automotive data and serves real-time predictions via a production-ready Flask interface deployed in the cloud.

👤 Developer Profile
Name: ARAVAPALLI M ADITYA SRIRAM
Course: Integrated Mtech in Artificial Intelligence
🔗 Project Links
Live Web Application: [https://car-price-prediction-app-00au.onrender.com](https://car-price-predictor-4ppt.onrender.com)
GitHub Repository: [https://github.com/AkshatGarg2005/car-price-prediction-app](https://github.com/itzmeadi/CAR-PRICE-PREDICTOR)
🛠️ Tech Stack & Ecosystem
Core Language: Python 3.10
Machine Learning & Data Processing: Scikit-learn, Pandas, NumPy
Backend Web Framework: Flask
Production Web Server: Gunicorn (WSGI)
Model Serialization: Pickle
Cloud Platform: Render (Virginia Region Gateway)
📂 Production Directory Architecture
The repository is engineered according to institutional production guidelines, strictly separating static client UI assets from data matrix operations:

📁 car-price-prediction-app/
│
├── 📁 static/
│   └── 📄 style.css            # Custom forest green user interface theme
│
├── 📁 templates/
│   └── 📄 index.html           # Core frontend interactive web form
│
├── 📄 app.py                   # Production Flask application engine
├── 📄 train.py                 # Automated pipeline download & ML model trainer
├── 📄 car_price_model.pkl      # Serialized Random Forest Regressor binary
├── 📄 requirements.txt         # Plaintext manifest mapping core dependencies
├── 📄 Procfile                 # Production WSGI process container config
└── 📄 .gitignore               # Excludes runtime caches and heavy dataset packages

📊 Dataset & Feature Alignment
The machine learning pipeline utilizes the Vehicle Dataset from CarDekho (sourced dynamically from Kaggle). The following features are processed, mapped into numerical fields, and evaluated by the mathematical boundaries of the model:

Feature Name	Type	Description / Mapped Space
Present Price	Continuous	Current showroom valuation of the vehicle (in Lakhs)
Kilometers Driven	Discrete	Total numeric mileage history of the car
Fuel Type	Categorical	Mapped Target Vectors: Petrol: 0, Diesel: 1, CNG: 2
Seller Type	Categorical	Mapped Target Vectors: Dealer: 0, Individual: 1
Transmission	Categorical	Mapped Target Vectors: Manual: 0, Automatic: 1
Owners History	Discrete	Integer count of previous registered owners
Car Age	Continuous	Transformed temporal matrix calculated from the build year
⚙️ How to Setup and Run Locally
Follow these instructions to establish an isolated local environment and execute the application:

1. Environment Setup & Component Installation
Open your terminal inside the root directory and activate your target environment manager:

# Create an isolated Conda workspace environment
conda create -n carprice python=3.10 -y
conda activate carprice

# Install the exact production ecosystem dependencies
pip install -r requirements.txt
2. Run the Machine Learning Pipeline
If you wish to re-train the underlying architecture and build the serialized prediction file, invoke the pipeline execution script:

python train.py
3. Launch the Local Development Engine
Initiate the Flask server microkernel locally:

python app.py
Once initialized, navigate to http://127.0.0.1:5000 inside your browser to interact with the application locally.

🚀 Live Cloud Deployment
The architecture is configured for continuous delivery via Git tracking connected directly to a cloud infrastructure cluster hosted on Render:

The production server boots up efficiently using a standard Linux container layer execution guided by the entry rules defined inside the project Procfile.
All connections parse clean POST form request arrays, routing calculations safely away from client viewpoints directly through the cloud-based car_price_model.pkl matrix without any hardcoded fallback conditions.
