  
import streamlit as st  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  

st.title("Quiz STA - Data Analysis")

# Sidebar untuk nama kelompok
st.sidebar.title("Team Member")
st.sidebar.markdown("""
- "Salsabila Hidayat"  
- "Wijaya Putra Sochinibe Lahagu"
- "Iren Samantha Priyani Zebua"
""")

# Load dataset
file_path = "selamat.csv"
df = pd.read_csv(file_path, header=None) 

# Ensure all data is displayed
pd.set_option('display.max_rows', None)  
pd.set_option('display.max_columns', None)  

st.title("Data Quiz App")
st.write("### Dataset Overview")
st.write(df)

# Display descriptive statistics
st.write("### Descriptive Statistics")
st.write(df.describe())

# Pilih hanya kolom numerik
df_numeric = df.select_dtypes(include=['number'])

# Pilih 5 kolom pertama agar lebih ringan
selected_cols = df_numeric.columns[:5]

fig, ax = plt.subplots(figsize=(12, 6))
for col in selected_cols:
    # Buat histogram
    counts, bin_edges = np.histogram(df_numeric[col].dropna(), bins=10)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.plot(bin_centers, counts, marker="o", linestyle="-", label=col)
ax.legend()
ax.set_title("Polygon Frequency of Selected Columns")
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.grid(True)
st.pyplot(fig)

# Boxplot untuk 5 kolom pertama
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df_numeric.iloc[:, :5], ax=ax)
ax.set_title("Boxplot of Selected Columns")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.grid(True)
st.pyplot(fig)

# Heatmap korelasi
if df_numeric.shape[1] > 20:
    df_numeric = df_numeric.iloc[:, :20]  # Batasi 20 kolom agar cepat
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df_numeric.corr(method="spearman"), cmap="coolwarm", linewidths=0.5, square=True, ax=ax)
ax.set_title("Optimized Heatmap Correlation", fontsize=12, fontweight="bold")
st.pyplot(fig)

# Population vs Sample Distribution
st.write("### Population vs Sample Distribution")
sample_size = int(0.2 * len(df_numeric))
df_sample = df_numeric.sample(sample_size, random_state=42)

fig, ax = plt.subplots(figsize=(12, 6))
for column in df_numeric.columns:
    sns.kdeplot(df_numeric[column], label=f"Population - {column}", fill=True, alpha=0.4, ax=ax)
    sns.kdeplot(df_sample[column], label=f"Sample - {column}", linestyle="--", ax=ax)
ax.set_xlabel("Value")
ax.set_ylabel("Density")
st.pyplot(fig)

