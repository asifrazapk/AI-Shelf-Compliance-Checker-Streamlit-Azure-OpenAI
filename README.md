# 📊 Planogram Compliance AI (Streamlit + Azure OpenAI)

An AI-powered application to **analyze retail shelf compliance** by comparing a **planogram (expected layout)** with an **actual shelf image** using Vision + LLM.

---

## 🚀 Features

- 📋 Extract products from **Planogram images**
- 📷 Extract products from **Real Shelf images**
- 🔍 Perform **fuzzy matching comparison**
- 📊 Generate **compliance dashboard**
- 📈 Shelf-wise insights & visualizations
- 🧾 Export all intermediate + final JSON outputs
- ⚡ Real-time progress tracking

---

## 🧠 How It Works (3-Step Pipeline)

### 1️⃣ Planogram Extraction
- Reads structured shelf layout
- Outputs products shelf-by-shelf

### 2️⃣ Actual Shelf Extraction
- Detects real-world products from image
- Handles blurry / noisy text

### 3️⃣ Compliance Comparison
- Fuzzy matching (≈70% similarity)
- Detects:
  - ✅ Matches
  - ❌ Missing products
  - 🔄 Replacements
  - ⚠️ Unexpected products

---

## 📊 Output Example

```json
{
  "overall_compliance": {
    "matching_percentage": "78%",
    "non_matching_percentage": "22%",
    "confidence": 85
  }
}
```

---

## 🛠 Tech Stack

- Streamlit  
- Azure OpenAI  
- RapidFuzz  
- Plotly  
- Python  

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 👨‍💻 Author

Asif

---

## 📜 License

MIT
