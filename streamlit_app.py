  
import streamlit as st  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  

file_path = "selamat.csv"
df = pd.read_csv(file_path, header=None) 

# Ensure all data is displayed
pd.set_option('display.max_rows', None)  
pd.set_option('display.max_columns', None)  

# Display entire dataset overview
print("Dataset Overview:")
print(df) 

# Display descriptive statistics
print("
Descriptive Statistics:")
print(df.describe())

plt.figure(figsize=(12, 6))

# Pilih 5 kolom pertama agar lebih ringan
selected_cols = df_numeric.columns[:5]

for col in selected_cols:
    # Buat histogram dulu
    counts, bin_edges = np.histogram(df_numeric[col], bins=10)

    # Ambil titik tengah setiap bin
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot sebagai garis (Polygon Frequency)
    plt.plot(bin_centers, counts, marker="o", linestyle="-", label=col)

plt.legend()
plt.title("Polygon Frequency of Selected Columns")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(data=df_numeric.iloc[:, :5])  # Pilih 5 kolom pertama agar tidak berat
plt.title("Boxplot of Selected Columns")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Pilih hanya kolom numerik
df_numeric = df.select_dtypes(include=['number'])

# Buat boxplot untuk seluruh dataset
plt.figure(figsize=(15, 6))
sns.boxplot(data=df_numeric)
plt.xticks(rotation=90)  # Rotasi label sumbu X biar rapi
plt.title("Boxplot of Entire Dataset")
plt.grid(True)
plt.show()

# Pilih hanya kolom numerik & sampling kalau terlalu besar
df_numeric = df.select_dtypes(include=['number'])

# Kurangi dimensi kalau dataset terlalu besar (ambil max 20 kolom biar cepat)
if df_numeric.shape[1] > 20:
    df_numeric = df_numeric.iloc[:, :20]

# Buat heatmap lebih cepat
plt.figure(figsize=(10, 6))
sns.heatmap(df_numeric.corr(method="spearman"),  # Bisa ganti ke 'kendall' buat lebih cepat
            annot=False,  # Matikan angka kalau lemot
            cmap="coolwarm",
            linewidths=0.5,
            square=True)

# Tambahkan judul
plt.title("Optimized Heatmap Correlation", fontsize=12, fontweight="bold")

plt.show()

st.title("Population vs Sample Distribution")

# Visualisasi
df_numeric = df.select_dtypes(include=['number'])
sample_size = int(0.2 * len(df_numeric))
df_sample = df_numeric.sample(sample_size, random_state=42)

fig, ax = plt.subplots(figsize=(12,6))
for column in df_numeric.columns:
    sns.kdeplot(df_numeric[column], label=f"Population - {column}", fill=True, alpha=0.4, ax=ax)
    sns.kdeplot(df_sample[column], label=f"Sample - {column}", linestyle="--", ax=ax)
plt.xlabel("Value")
plt.ylabel("Density")
st.pyplot(fig)
