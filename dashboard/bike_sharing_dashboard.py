import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set style for seaborn
sns.set(style='dark')

# Fungsi untuk membuat DataFrame harian
def create_daily_orders_df(df):
    daily_orders_df = df.resample(rule='D', on='dteday').agg({
        "cnt": "sum"  # Use 'cnt' for total rentals
    })
    daily_orders_df = daily_orders_df.reset_index()
    daily_orders_df.rename(columns={"cnt": "total_rentals"}, inplace=True)
    
    return daily_orders_df

# Fungsi untuk menghitung rata-rata penyewaan per jam
def create_hourly_rentals_df(df):
    hourly_rentals_df = df.groupby(df['dteday'].dt.hour).agg({"cnt": "sum"}).reset_index()
    hourly_rentals_df.rename(columns={"cnt": "count"}, inplace=True)
    return hourly_rentals_df

# Load data dari CSV
hour_data = pd.read_csv("hour_data_cleaned.csv")
hour_data['dteday'] = pd.to_datetime(hour_data['dteday']) 

# Mengurutkan DataFrame
hour_data.sort_values(by="dteday", inplace=True)
hour_data.reset_index(drop=True, inplace=True)

# Komponen Filter
min_date = hour_data["dteday"].min()
max_date = hour_data["dteday"].max()

# Membuat daftar tanggal penting
important_dates = ['2012-10-30']

# Menambahkan kolom untuk menandai peristiwa penting
hour_data['event'] = np.where(hour_data['dteday'].isin(important_dates), 1, 0)

with st.sidebar:
    st.image("Saya.jpg")
    start_date, end_date = st.date_input(
        label='Rentang Waktu',
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

# Memfilter data
main_df = hour_data[(hour_data["dteday"] >= str(start_date)) & (hour_data["dteday"] <= str(end_date))]

# Membuat DataFrame untuk visualisasi
daily_orders_df = create_daily_orders_df(main_df)
hourly_rentals_df = create_hourly_rentals_df(main_df)

# Header Dashboard
st.header('Bike Sharing Dashboard :bike:')
st.subheader('Daily Rentals')

# Menampilkan total penyewaan
col1, col2 = st.columns(2)

with col1:
    total_rentals = daily_orders_df.total_rentals.sum()
    st.metric("Total Rentals", value=total_rentals)

with col2:
    st.metric("Total Days", value=len(daily_orders_df))

# Visualisasi Jumlah Penyewaan Harian
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(daily_orders_df["dteday"], daily_orders_df["total_rentals"], marker='o', linewidth=2, color="#90CAF9")
ax.set_title("Daily Rentals", fontsize=20)
ax.set_ylabel("Total Rentals", fontsize=15)
ax.set_xlabel("Date", fontsize=15)
st.pyplot(fig)

# Menambahkan informasi tambahan
avg_rentals_per_day = daily_orders_df["total_rentals"].mean()
st.write(f"Rata-rata penyewaan harian dalam rentang tanggal yang dipilih adalah {avg_rentals_per_day:.2f} sepeda per hari.")

# Visualisasi Penyewaan per Jam
st.subheader("Hourly Rentals")
hourly_rentals_df = main_df.groupby('hr')['cnt'].sum().reset_index()
hourly_rentals_df.rename(columns={'cnt': 'total_rentals'}, inplace=True)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x="hr", y="total_rentals", data=hourly_rentals_df, palette="viridis")
ax.set_title("Total Number of Rentals by Hour", fontsize=20)
ax.set_xlabel("Hour of the Day", fontsize=15)
ax.set_ylabel("Total Rentals", fontsize=15)
ax.set_xticks(range(0, 24))  # Show all hours from 0 to 23
ax.set_xticklabels([str(hour) + ":00" for hour in range(24)], rotation=45)
st.pyplot(fig)

# Menambahkan informasi tambahan
busiest_hour = hourly_rentals_df.loc[hourly_rentals_df["total_rentals"].idxmax()]["hr"]
st.write(f"Jam dengan penyewaan tertinggi adalah pada jam {busiest_hour}:00 dengan total penyewaan paling banyak.")

# ========== 1. ANALISIS MUSIMAN ========== #
st.subheader('Rata-rata Penyewaan Berdasarkan Musim')

# Menghitung rata-rata penyewaan berdasarkan musim
season_avg = hour_data.groupby('season')['cnt'].mean().reset_index()
season_avg['season'] = season_avg['season'].map({1: 'Musim Semi', 2: 'Musim Panas', 3: 'Musim Gugur', 4: 'Musim Dingin'})

# Visualisasi rata-rata penyewaan berdasarkan musim
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=season_avg, x='season', y='cnt', palette='viridis', ax=ax)
ax.set_title('Rata-rata Penyewaan Sepeda Berdasarkan Musim')
ax.set_ylabel('Rata-rata Penyewaan')
ax.set_xlabel('Musim')
st.pyplot(fig)

# Menambahkan informasi tambahan
season_highest_rentals = season_avg.loc[season_avg['cnt'].idxmax()]['season']
avg_rentals_by_season = season_avg['cnt'].max()
st.write(f"Musim dengan rata-rata penyewaan tertinggi adalah {season_highest_rentals}, dengan rata-rata {avg_rentals_by_season:.2f} penyewaan.")

# ========== 2. PENGARUH CUACA ========== #
st.subheader('Pengaruh Variabel Cuaca terhadap Penyewaan')

# Visualisasi pengaruh cuaca
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot Suhu vs Penyewaan
sns.scatterplot(data=hour_data, x='temp', y='cnt', color='blue', ax=axs[0])
axs[0].set_title('Pengaruh Suhu terhadap Jumlah Penyewaan')
axs[0].set_xlabel('Suhu (Normalisasi)')
axs[0].set_ylabel('Jumlah Penyewaan')

# Scatter plot Kelembapan vs Penyewaan
sns.scatterplot(data=hour_data, x='hum', y='cnt', color='orange', ax=axs[1])
axs[1].set_title('Pengaruh Kelembapan terhadap Jumlah Penyewaan')
axs[1].set_xlabel('Kelembapan (Normalisasi)')
axs[1].set_ylabel('Jumlah Penyewaan')

st.pyplot(fig)

# Menambahkan informasi tambahan
corr_temp = hour_data['temp'].corr(hour_data['cnt'])
corr_hum = hour_data['hum'].corr(hour_data['cnt'])
st.write(f"Korelasi antara suhu dan jumlah penyewaan adalah {corr_temp:.2f}, sedangkan korelasi antara kelembapan dan jumlah penyewaan adalah {corr_hum:.2f}.")

# ========== 3. SEGMENTASI PENGGUNA ========== #
st.subheader('Total Penyewaan Berdasarkan Tipe Pengguna')

# Menghitung total penyewaan berdasarkan tipe pengguna
user_segmentation = hour_data[['casual', 'registered']].sum().reset_index()
user_segmentation.columns = ['Tipe Pengguna', 'Total Penyewaan']

# Visualisasi total penyewaan berdasarkan tipe pengguna
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=user_segmentation, x='Tipe Pengguna', y='Total Penyewaan', palette='pastel', ax=ax)
ax.set_title('Total Penyewaan Sepeda Berdasarkan Tipe Pengguna')
ax.set_ylabel('Total Penyewaan')
ax.set_xlabel('Tipe Pengguna')
st.pyplot(fig)

# Menambahkan informasi tambahan
casual_rentals = user_segmentation[user_segmentation['Tipe Pengguna'] == 'casual']['Total Penyewaan'].values[0]
registered_rentals = user_segmentation[user_segmentation['Tipe Pengguna'] == 'registered']['Total Penyewaan'].values[0]
st.write(f"Pengguna kasual menyewa total {casual_rentals} sepeda, sedangkan pengguna terdaftar menyewa total {registered_rentals} sepeda.")


# ========== 4. MODEL PREDIKSI ========== #
st.subheader('Hasil Prediksi vs. Nilai Aktual')

# Memisahkan data menjadi fitur dan target
X = hour_data[['temp', 'hum', 'windspeed', 'season', 'holiday', 'weekday']]
y = hour_data['cnt']

# One-hot encoding untuk variabel kategorikal
X = pd.get_dummies(X, columns=['season', 'holiday', 'weekday'], drop_first=True)

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat dan melatih model regresi
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Melakukan prediksi
y_pred = reg_model.predict(X_test)

# Visualisasi hasil prediksi vs nilai aktual
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_test, y_pred, color='purple', alpha=0.5)
ax.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--')  # Garis ideal
ax.set_title('Hasil Prediksi vs. Nilai Aktual')
ax.set_xlabel('Nilai Aktual')
ax.set_ylabel('Nilai Prediksi')
st.pyplot(fig)

# Menampilkan MSE dan R²
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f'Mean Squared Error: {mse}')
st.write(f'R² Score: {r2}')

# Menambahkan kolom untuk menandai peristiwa penting
hour_data['event'] = np.where(hour_data['dteday'].isin(important_dates), 1, 0)

# ========== 5. VISUALISASI PENYEWAAN BERDASARKAN PERISTIWA ========== #
st.subheader('Jumlah Penyewaan Berdasarkan Peristiwa Penting')

# Visualisasi penyewaan berdasarkan peristiwa
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=hour_data, x='dteday', y='cnt', hue='event', palette='coolwarm', ax=ax)
ax.set_title('Jumlah Penyewaan Sepeda Berdasarkan Peristiwa Penting')
ax.set_xlabel('Tanggal')
ax.set_ylabel('Jumlah Penyewaan')
plt.xticks(rotation=45)
st.pyplot(fig)

# Menambahkan informasi tambahan
event_avg = hour_data[hour_data['event'] == 1]['cnt'].mean()
normal_avg = hour_data[hour_data['event'] == 0]['cnt'].mean()
st.write(f"Rata-rata penyewaan pada hari biasa dalam rentang tanggal yang dipilih adalah {normal_avg:.2f}, "f"sedangkan pada peristiwa penting adalah {event_avg:.2f}.")

# ========== 6. VISUALISASI PENYEWAAN BERDASARKAN TIPE PENGGUNA ========== #
st.subheader('Jumlah Penyewaan Berdasarkan Tipe Pengguna Selama 24 Jam')

# Menghitung total penyewaan berdasarkan tipe pengguna
user_counts = hour_data.groupby('hr')[['casual', 'registered']].sum().reset_index()
user_counts['total'] = user_counts['casual'] + user_counts['registered']

# Visualisasi penyewaan per jam berdasarkan tipe pengguna
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=user_counts, x='hr', y='casual', label='Pengguna Kasual', color='blue', ax=ax)
sns.lineplot(data=user_counts, x='hr', y='registered', label='Pengguna Terdaftar', color='orange', ax=ax)
ax.set_title('Jumlah Penyewaan Berdasarkan Tipe Pengguna Selama 24 Jam')
ax.set_xlabel('Jam')
ax.set_ylabel('Jumlah Penyewaan')
plt.xticks(range(0, 24))
st.pyplot(fig)

# Menambahkan informasi tambahan
total_casual = user_counts['casual'].sum()
total_registered = user_counts['registered'].sum()
dominant_user_type = 'Pengguna Kasual' if total_casual > total_registered else 'Pengguna Terdaftar'
st.write(f"Dalam rentang waktu yang dipilih, {dominant_user_type} lebih banyak menyewa sepeda.")

# ========== 7. RATA-RATA PENYEWAAN: PERISTIWA PENTING VS HARI BIASA ========== #
st.subheader('Perbandingan Rata-rata Penyewaan: Hari Biasa vs Peristiwa Penting')

# Menyusun data untuk visualisasi rata-rata penyewaan
avg_comparison = pd.DataFrame({
    'Event': ['Hari Biasa', 'Peristiwa Penting'],
    'Rata-rata Penyewaan': [normal_avg, event_avg]
})

# Visualisasi perbandingan rata-rata penyewaan
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=avg_comparison, x='Event', y='Rata-rata Penyewaan', palette='muted', ax=ax)
ax.set_title('Perbandingan Rata-rata Penyewaan: Hari Biasa vs Peristiwa Penting')
ax.set_xlabel('Jenis Hari')
ax.set_ylabel('Rata-rata Penyewaan')
st.pyplot(fig)

# Menambahkan informasi tambahan
st.write(f"Pada rentang waktu yang dipilih, rata-rata penyewaan pada hari biasa adalah {normal_avg:.2f}, "f"sedangkan pada peristiwa penting rata-rata penyewaan adalah {event_avg:.2f}.")

# Caption
st.caption('M Dicky Desriansyah')
