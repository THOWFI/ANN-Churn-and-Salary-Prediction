# 🧠 ANN — Churn & Salary Prediction (Streamlit Deployments)

A lightweight **ANN-based** repo containing two Streamlit apps:

- **`app.py`** — Customer **Churn** Prediction (classification; Keras ANN)  
- **`rgapp.py`** — Customer **Salary** Prediction (regression; Keras ANN)

This README is created specifically from the repository you provided:
`https://github.com/THOWFI/ANN-Churn-and-Salary-Prediction` (and the uploaded zip).  
It documents the actual files, models and pickles in the repo, how to run locally, and deployment notes (you mentioned it is already deployed on Streamlit — add your share URL below).

---

## ✅ Highlights
- **Two production-ready Streamlit apps** for inference using pre-trained Keras models (`.h5`)
- **Preprocessing artifacts included**: OneHotEncoders, LabelEncoders and StandardScalers (pickled)
- **All required notebooks & logs** included for retraining / experiments
- **Easy local run** with `streamlit run app.py` / `streamlit run rgapp.py`
- **Deployed to Streamlit** (add your Streamlit share URL below)

---

## 📦 Project Tree (Core)

ANN-Churn-and-Salary-Prediction/
│── .gitignore
│── Churn_Modelling.csv # raw dataset used for churn model
│── app.py # Streamlit app — Churn prediction (classification)
│── rgapp.py # Streamlit app — Salary prediction (regression)
│── model.h5 # pre-trained churn model (Keras .h5)
│── rgmodel.h5 # pre-trained salary model (Keras .h5)
│── ohe_geo.pkl # OneHotEncoder (geography) — churn preprocessing
│── ohe_geo_salary.pkl # OneHotEncoder (geography) — salary preprocessing
│── le_gender.pkl # LabelEncoder (gender) — churn preprocessing
│── le_gender_salary.pkl # LabelEncoder (gender) — salary preprocessing
│── scaler.pkl # StandardScaler — churn preprocessing
│── scaler_salary.pkl # StandardScaler — salary preprocessing
│── experiments.ipynb # training/EDA notebook(s)
│── prediction.ipynb # example inference notebook
│── logs/ # TensorBoard logs (training runs)
│── requirements.txt # Python dependencies
└── README.md 

---

> **Deployed:** Done in Streamlit deployment using GitHub Repo

---

## 🧩 What’s inside / Core ML pieces

### Data
- `Churn_Modelling.csv` — main dataset used for training the churn ANN (contains columns such as `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`, `Exited`).

### Preprocessing artifacts (pickles)
- `ohe_geo*.pkl` — OneHotEncoder(s) for Geography (France/Germany/Spain)
- `le_gender*.pkl` — LabelEncoder(s) for Gender
- `scaler*.pkl` — StandardScalers used to normalize numeric features

These artifacts must be loaded by the Streamlit app to encode inputs exactly as training.

### Models
- `model.h5` — churn classifier (Keras) — outputs probability (used threshold = 0.5)
- `rgmodel.h5` — salary regressor (Keras) — outputs numeric salary estimate

### Notebooks / Logs
- `experiments.ipynb`, `prediction.ipynb` — for EDA, training, testing and example inference
- `logs/` — TensorBoard logs (training history visible via tensorboard)

---

## 🔑 Environment & Requirements

Local `.env` / secrets: **Not required** for local inference apps (apps load local files).  
Make sure you have the following installed (or pin versions in `requirements.txt`):

Contents of `requirements.txt` (as included in repo):

pandas
numpy
seaborn
matplotlib
scikit-learn
tensorflow
tensorboard
streamlit
ipykernel


**Recommended notes**
- Use a compatible TensorFlow version for the saved `.h5` models (if you trained with TF 2.x, use the matching 2.x release).
- On low-memory / CPU-only machines: consider `pip install tensorflow-cpu`.
- If you get pickle/unpickle issues, ensure `numpy` and `scikit-learn` versions are compatible with the pickled objects.

---

## 🧭 Config Checks Before Running

Open the repo files and confirm:
- `model.h5` and `rgmodel.h5` are present in the repo root.
- The pickles `ohe_geo.pkl`, `le_gender.pkl`, `scaler.pkl` (and their salary variants) are present.
- `requirements.txt` installed into your environment.
- If you retrain locally, ensure the **feature order** used when saving the scaler is preserved when building input data for inference.

---

## 🧪 Quick Start (Local)

```bash
# 1) Clone repository
git clone https://github.com/THOWFI/ANN-Churn-and-Salary-Prediction.git
cd ANN-Churn-and-Salary-Prediction

# 2) Create & activate venv
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4) Run the Streamlit apps
streamlit run app.py       # Churn prediction app
streamlit run rgapp.py     # Salary prediction app

**Streamlit Deployment**

* To deploy on Streamlit Cloud (share.streamlit.io), push repo to GitHub and follow the Streamlit share instructions.
* Add the repo & branch in Streamlit sharing settings and deploy.
* Paste the generated share URL into this README in the `Deployed` field above.
````

---

## 🛠️ How the Streamlit apps work (high-level)

**`app.py` (Churn)**

1. Loads `model.h5`, `ohe_geo.pkl`, `le_gender.pkl`, `scaler.pkl`.
2. Builds a user input form (Geography, Gender, CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary).
3. Encodes categorical features using saved encoders (label + OHE), aligns columns with the scaler order.
4. Applies `scaler.transform(...)` then `model.predict(...)`.
5. Displays churn probability and a simple thresholded message (prob > 0.5 → churn likely).

**`rgapp.py` (Salary)**

1. Loads `rgmodel.h5`, `ohe_geo_salary.pkl`, `le_gender_salary.pkl`, `scaler_salary.pkl`.
2. Accepts user inputs (features used in training).
3. Encodes & scales the inputs, calls the regressor, and displays estimated salary.

---

## 🧾 Example (how to use apps)

1. `streamlit run app.py`

   * Fill the form fields, click **Predict** — app will return "Churn probability: 0.XX" and a message.
2. `streamlit run rgapp.py`

   * Fill the form fields, click **Estimate Salary** — app will show the predicted salary.

---

## 🧩 Stage Notes (how this repo was structured)

* **Data**: `Churn_Modelling.csv` used for training / EDA (not required for inference if models are present).
* **Preprocessing**: Encoders (OHE/Label) and scalers were fitted and saved as pickles.
* **Modeling**: ANN models trained using Keras (TensorFlow backend) and saved as `.h5` files.
* **Serving**: Streamlit apps load the saved model + preprocessing artifacts and perform inference.

---

## 🔮 Notes on Retraining & Experiments

* Retrain using the included notebooks (`experiments.ipynb`) — they contain the training pipeline and `model.fit()` calls and log to `logs/` (TensorBoard).
* After retraining, re-save:

  * `model.h5` / `rgmodel.h5`
  * updated `scaler*.pkl`, `ohe_geo*.pkl`, `le_gender*.pkl`

---

## ☁️ (Optional) Deployment Tips

You already mentioned the apps are deployed on Streamlit. If you want alternative production paths:

* Dockerize the Streamlit app(s) and deploy on a VM or container service.
* Wrap the model + preprocessing into a single `skorch`/`tf.saved_model` pipeline and serve via FastAPI + Uvicorn (for production-grade APIs).
* Use Streamlit Cloud for simple, interactive deployments (free tier available).

---

## 📜 License

For **educational and research** purposes. Validate and test thoroughly before using in production.
