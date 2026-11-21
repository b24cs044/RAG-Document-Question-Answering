# RAG-Based Document Question Answering System

## 📂 Document Setup

The original PDF documents used during development are not included in this GitHub repository. Users must upload their own documents to Google Drive before running the project.

### Steps to Upload Documents

1. Open [Google Drive](https://drive.google.com/).
2. Create a folder named `RAG_Project`.
3. Inside `RAG_Project`, create another folder named `documents`.
4. Upload your PDF documents into the `documents` folder.

Your Google Drive folder structure should look like this:

```text
MyDrive/
└── RAG_Project/
    └── documents/
        ├── document1.pdf
        ├── document2.pdf
        └── document3.pdf
```

5. Open `RAG_Document_QA_Project.ipynb` in Google Colab.
6. Run the notebook cells in the correct order.
7. When prompted, allow Google Colab to access your Google Drive.

**Important:** The notebook expects PDF files in the following location:

```python
/content/drive/MyDrive/RAG_Project/documents
```

If you use a different folder location, update the `DOCUMENT_DIR` variable in the notebook accordingly.

The system will automatically extract text from the uploaded PDFs, split the text into chunks, generate embeddings, and store them in ChromaDB for retrieval.


## 1. Project Overview

This project implements a Retrieval-Augmented Generation
(RAG) system that answers questions using uploaded PDF
documents.

It combines semantic search, vector databases,
embeddings, and Large Language Models.

## 2. Technologies Used

- Python
- Google Colab
- ChromaDB
- Sentence Transformers
- LangChain Text Splitters
- Google Gemini API
- Gradio

## 3. Document Collection

The project uses four PDF documents:

1. ATM_Code_Explanation.pdf
2. ATM_GUI_Presentation_Guide.pdf
3. Fall_Detection_Project_Guide.pdf
4. Fall_Detection_Graph_Explanations.pdf

Total pages: 55.

## 4. System Architecture

PDF Documents
     |
     v
Text Extraction
     |
     v
Text Chunking
     |
     v
Embedding Generation
     |
     v
ChromaDB Vector Database
     |
     v
Semantic Retrieval
     |
     v
Retrieved Context + User Question
     |
     v
Gemini LLM
     |
     v
Grounded Answer

## 5. Chunking Configuration

- Chunk size: 350 tokens
- Chunk overlap: 60 tokens
- Embedding model: all-mpnet-base-v2

## 6. Vector Database

ChromaDB is used to store document chunks,
embeddings, and metadata.

Similarity search retrieves the most relevant
chunks for each user question.

## 7. RAG vs Without RAG

======================================================================
QUESTION:
What is the exact name of the function used for Storing hash-based authentication in the databasedescribed in my ATM code PDF?

======================================================================
WITHOUT RAG:
I do not know the answer. I do not have access to your ATM code PDF or any specific documents, so I cannot tell you the exact name of the function used in your project.

======================================================================
WITH RAG:
Based on the provided context, the exact name of the function used for hashing PINs for hash-based authentication is **`hash_pin(pin)`** (or `hash_pin`). 

It turns a PIN into a 64-character SHA-256 hash so that only the hash is stored in the database.

**Source:**
* `ATM_Code_Explanation.pdf`, Page 2

======================================================================
RETRIEVED SOURCES:
ATM_Code_Explanation.pdf | Page: 6
ATM_Code_Explanation.pdf | Page: 2
ATM_Code_Explanation.pdf | Page: 2
ATM_GUI_Presentation_Guide.pdf | Page: 2
ATM_GUI_Presentation_Guide.pdf | Page: 5

## 8. Test Result for multiple questions

======================================================================
BATCH RAG RESULTS
======================================================================
### QUESTION 1: What is the main purpose of the ATM application?

**Answer:**
The provided context does not contain information stating the main purpose of the ATM application. 

*Sources referenced: ATM_GUI_Presentation_Guide.pdf (Pages 1, 5, 6), ATM_Code_Explanation.pdf (Pages 6, 7)*

---

### QUESTION 2: Which functions are described in the ATM code explanation?

**Answer:**
The following functions are described in the *ATM Code Explanation* document:

1. **`logic.login()`**: Fetches an account using a parameterized `SELECT` query, checks if the account is locked, and compares the SHA-256 hash of the entered PIN against the stored hash. Increments `failed_attempts` on failure or resets it to 0 on success (*ATM_Code_Explanation.pdf*, Page 6).
2. **`withdraw()`**: Reads the account balance and verifies whether `amount > balance`. If true, raises `ATMError` to prevent updating the database or creating a transaction record. Subtracts the amount, records a withdraw row, and commits (*ATM_Code_Explanation.pdf*, Pages 2, 6).
3. **`transfer(from, to, amount)`**: Validates the recipient, subtracts funds from one account and adds to another, records two transaction rows, and commits once (*ATM_Code_Explanation.pdf*, Page 2).
4. **`create_account(...)`**: Validates input fields, hashes the PIN, and inserts the new account into the database. Catches SQLite `IntegrityError` if a duplicate primary key (`acc_no`) is submitted (*ATM_Code_Explanation.pdf*, Pages 2, 7).
5. **`change_pin(acc, old, new)`**: Ensures the new PIN is 4 digits, checks that the old PIN matches, and saves the new hash (*ATM_Code_Explanation.pdf*, Page 2).
6. **`_record()`**: Inserts transaction details (type, amount, `datetime.now()` timestamp, and updated balance) into the `transactions` table within the same database commit (*ATM_Code_Explanation.pdf*, Page 6).

*Sources referenced: ATM_Code_Explanation.pdf, Pages 2, 6, 7*

---

### QUESTION 3: How does the fall detection project process sensor data?

**Answer:**
The fall detection project processes sensor data through the following steps:

1. **Segmentation & Windowing**: Sensor data is grouped into continuous segments (sharing subject, activity, and class) and sliced into fixed 200-sample windows (representing 2 seconds at 100 Hz) (*Fall_Detection_Project_Guide.pdf*, Pages 18, 20).
2. **Impact Detection & Centering**: 
   - Fall windows are centered around an acceleration magnitude impact peak (`mag = X[:, 6]`).
   - The threshold (`thr`) for an impact is calculated using the 99th percentile of magnitude during normal activities (`cfg["fall_peak_pctl"]`). If no spike exceeds the threshold, it falls back to the peak point (`np.argmax(m)`) (*Fall_Detection_Project_Guide.pdf*, Pages 20, 29).
   - Augmentation and position invariance are achieved by shifting the impact window using five offset positions (−50 to +50 samples) (*Fall_Detection_Project_Guide.pdf*, Page 29).
3. **Feature Extraction (for Tree Models)**:
   - `extract_features(X)` flattens each 200-sample, 8-channel window into 92 features using `np.hstack(F)` (*Fall_Detection_Project_Guide.pdf*, Page 23).
   - Calculates statistics over time (mean, std, min, max, median, range via `np.ptp`, energy, skew, kurtosis, and zero-crossing rate `zcr`).
   - Calculates Jerk (max and mean absolute changes between neighboring samples for Ax, Ay, Az, and magnitude).
   - Extracts 4 additional features: Signal Magnitude Area (SMA) and the standard deviation of acceleration magnitude across the first half, second half, and last 50 samples of the window (*Fall_Detection_Project_Guide.pdf*, Page 23).

*Sources referenced: Fall_Detection_Project_Guide.pdf, Pages 18, 20, 23, 29*

---

### QUESTION 4: What do the graphs in the fall detection guide represent?

**Answer:**
The graphs in the guide represent performance, feature importances, and diagnostic metrics across model evaluations:

1. **Plot 2 (ROC and PR Curves)**: Displays Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves showing true/false positive rates across all thresholds along with AUC and Average Precision scores (*Fall_Detection_Project_Guide.pdf*, Page 25).
2. **Plot 3 (Probability Histogram)**: Displays log counts of predicted fall probabilities for real ADL vs. real fall windows, featuring a dashed vertical line indicating the chosen decision threshold (*Fall_Detection_Project_Guide.pdf*, Page 25).
3. **Plot 4 (Threshold Sweep)**: Plots precision, recall, and F1-score across threshold values ranging from 0.01 to 0.99 on the test set (*Fall_Detection_Project_Guide.pdf*, Page 25).
4. **Plot 5 (Error Analysis Graphs)**: Plots error rates per activity/true class for the 15 worst activities, as well as subject-level metrics including fall recall, false alarm rate, and window counts (*Fall_Detection_Project_Guide.pdf*, Page 25).
5. **Plot 6 (Example Windows & Attention Weights)**: Plots sample signal windows for True Positives (TP), False Negatives (FN), and False Positives (FP), as well as attention weights displayed as a red shaded area stretched over the 200-sample window to highlight focus regions (*Fall_Detection_Project_Guide.pdf*, Page 25).
6. **Plot 7 (Parameter Table)**: Outlines layer names, shapes, and parameter counts for neural network models (*Fall_Detection_Project_Guide.pdf*, Page 25).
7. **Random Forest Top 20 Feature Importances**: Ranks the top 20 hand-crafted features by their contribution to impurity reduction across all trees (e.g., `Ay_mean`, `GyroMag_range`) (*Fall_Detection_Graph_Explanations.pdf*, Page 3).

*Sources referenced: Fall_Detection_Project_Guide.pdf, Page 25; Fall_Detection_Graph_Explanations.pdf, Page 3*

---

### QUESTION 5: What specific implementation detail is described in the project guide?

**Answer:**
The project guide outlines several specific code implementation details:

1. **Multi-Seed Evaluation (Section 3.11b)**:
   - Uses a `BUILD` mapping dictionary from model names to classes.
   - Loops over seeds 1, 2, and 3 for 1D-CNN and CNN-LSTM models with `quiet=True` during training.
   - Tunes thresholds on validation data, evaluates test data, and calculates average and spread using `S.groupby("model")[...].agg(["mean", "std"])` (*Fall_Detection_Project_Guide.pdf*, Page 25).
2. **Directory & File Saving (Section 3.11c)**:
   - Ensures output directories exist via `os.makedirs(out, exist_ok=True)`.
   - Checks file presence with `os.path.exists` inside a loop before copying to prevent errors from skipped models (*Fall_Detection_Project_Guide.pdf*, Page 25).
3. **Libraries & Tools Used**:
   - Uses `find_peaks` for peak detection, `skew` and `kurtosis` for feature engineering, `RandomForestClassifier` and `xgboost` for tree baselines, and `sns.set_theme(style="whitegrid")` for plot styling (*Fall_Detection_Project_Guide.pdf*, Page 18).

*Sources referenced: Fall_Detection_Project_Guide.pdf, Pages 18, 25*




## 9. Evaluation

Number of test questions: 5

Correct answers: 5

Accuracy: 100%

## 10. How to Run

1. Open the Google Colab notebook.
2. Install the required dependencies.
3. Upload PDF documents to Google Drive.
4. Configure the Gemini API key in Colab Secrets.
5. Run the notebook cells sequentially.
6. Launch the Gradio interface.

## 11. Conclusion

The project demonstrates how RAG can improve
document-based question answering by retrieving
relevant information from a vector database
and providing it as context to an LLM.


