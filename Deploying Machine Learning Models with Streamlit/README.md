# 1️⃣ Create venv if needed
python -m venv .venv

# 2️⃣ Activate venv
# Windows:
.venv\Scripts\activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Train the model (one time)
python train_model.py

# 5️⃣ Run the Streamlit app
streamlit run app.py
