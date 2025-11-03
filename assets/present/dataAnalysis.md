# Analisis Deskriptif Dataset Botnet NCC-2

## 1. Karakteristik Data
Dataset **NCC-2 (Network Common Characterization Dataset)** merupakan dataset trafik jaringan yang digunakan untuk penelitian dalam bidang **Network Intrusion Detection System (NIDS)**, khususnya untuk **deteksi botnet, malware, dan serangan DDoS**.

**Ciri-ciri utama:**
- **Jumlah data:** ±65 juta aliran (flows) jaringan
- **Sumber data:** dikumpulkan dari jaringan nyata dan simulasi laboratorium  (CTU & ITS)
- **Tujuan:** Klasifikasi antara *benign (normal)* dan *malicious (botnet)* trafik
- **Label kelas:** 'Benign' dan 'Botnet' (misalnya Mirai, Bashlite, Torii, dll.)

---

## 2. Sifat Data
**Jenis dan sifat data NCC-2:**
- **Sifat:** Sekunder (dikumpulkan oleh pihak ketiga untuk riset publik)
- **Jenis data:** Kuantitatif (numerik) dan Kategorikal (protokol, label)
- **Tipe pengukuran:**
  - **Nominal:** 'Protocol', 'Label'
  - **Rasio/Interval:** 'Flow Duration', 'Pkt Len Mean', 'Tot Fwd Pkts'
- **Domain:** *Network Security / Intrusion Detection / Botnet Detection*
- **Keseimbangan data:** Tidak seimbang (*imbalanced*), benign (normalflow) jauh lebih banyak dari botnet

---

## 3. Bentuk Data
**Struktur data NCC-2:**
- **Format:** '.csv' atau '.arff' (tabular)
- **Baris:** representasi satu *flow* jaringan (satu sesi komunikasi)
- **Kolom:** atribut statistik dari aliran jaringan
- **Ukuran file:** dapat mencapai >10 GB tergantung subset
- **Sumber ekstraksi:** hasil *feature extraction* dari file PCAP menggunakan CICFlowMeter

---

## 4. Kesimpulan
> Dataset **Botnet NCC-2** merupakan dataset sekunder berbentuk tabular berisi rekaman aliran trafik jaringan. Data bersifat kuantitatif dengan sebagian besar fitur numerik dan label kategorikal. Analisis deskriptif menunjukkan distribusi tidak seimbang antara trafik normal dan botnet. Dataset ini sangat relevan untuk penelitian sistem deteksi intrusi berbasis machine learning.

---

## Workflow Stage

| **Tahap (Stage)**           | **Apa yang Dilakukan**                                                                                                                    | **Deskripsi Rinci**                                                                                                                                                                                                                       |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1️⃣ Data Loading**        | Membaca data dari file CSV melalui DuckDB untuk sensor tertentu.                                                                               | Tahap awal di mana sistem mengambil seluruh data jaringan (flow logs) dari sensor tertentu. Jika koneksi internal `getConnection()` gagal, sistem otomatis membuat koneksi lokal menggunakan DuckDB agar tetap dapat memuat data.         |
| **2️⃣ Preprocessing**       | Membersihkan data kosong, mengubah kolom kategori menjadi numerik, dan memetakan arah lalu lintas (`Dir`).                                     | Proses ini memastikan data siap untuk dianalisis oleh model. Kolom `Proto` dan `State` diubah menjadi angka menggunakan `LabelEncoder`, sedangkan kolom `Dir` dikonversi menjadi nilai numerik: `->` (1), `<-` (-1), dan `<->` (0).       |
| **3️⃣ Feature Engineering** | Membuat fitur tambahan seperti rasio, intensitas, dan keseimbangan untuk memperkuat pembelajaran model.                                        | Menambahkan fitur seperti `ByteRatio`, `DurationRate`, `FlowIntensity`, `TrafficBalance`, `DurationPerPkt`, dan `Intensity`. Fitur-fitur ini membantu model mengenali pola komunikasi abnormal yang biasa terjadi pada botnet.            |
| **4️⃣ Model**               | Melatih tiga model dasar (Random Forest, HistGradientBoosting, ExtraTrees) dan satu meta-learner XGBoost menggunakan metode stacking ensemble. | Teknik stacking menggabungkan kekuatan beberapa algoritma untuk meningkatkan akurasi prediksi. XGBoost dipilih sebagai meta-classifier karena performanya yang kuat terhadap data tidak seimbang dan non-linear.                          |
| **5️⃣ Thresholding**        | Menentukan ambang batas (threshold) terbaik berdasarkan optimisasi kurva ROC dengan metode G-Mean.                                             | Tahap ini mencari nilai threshold optimal yang menyeimbangkan antara sensitivitas (recall) dan spesifisitas (true negative rate), bukan sekadar menggunakan nilai default 0.5.                                                            |
| **6️⃣ Graph Construction**  | Membangun grafik jaringan di mana setiap IP menjadi simpul (node) dan setiap aliran data menjadi garis (edge).                                 | Data komunikasi antar-IP direpresentasikan sebagai graf. Seluruh node dipertahankan agar struktur jaringan tetap utuh, sementara edge dengan pasangan sumber-tujuan yang sama digabung (aggregated) agar visualisasi lebih efisien.       |
| **7️⃣ Auto Role Detection** | Mengklasifikasikan setiap node sebagai C&C, Bot, atau Normal berdasarkan arah lalu lintas dan probabilitas prediksi.                           | Sistem mendeteksi peran setiap IP secara otomatis berdasarkan perilaku komunikasi: node dengan inbound tinggi dan probabilitas besar → C&C; node outgoing dominan → Bot; sisanya → Normal. Tidak ada daftar IP tetap (purely behavioral). |
| **8️⃣ Visualization**       | Menampilkan grafik jaringan secara interaktif dengan warna sesuai peran setiap node.                                                           | Menggunakan Plotly untuk visualisasi dinamis. Node **C&C** ditampilkan berwarna merah, **Bot** oranye, dan **Normal** abu-abu. Setiap node menampilkan detail seperti `Probabilitas` dan `Inbound Ratio` saat di-hover.                   |
| **9️⃣ Export**              | Menyimpan hasil akhir dalam format HTML dan CSV untuk analisis lanjutan atau integrasi dashboard.                                              | Output akhir terdiri dari dua file: (1) **HTML interaktif** untuk visualisasi jaringan, dan (2) **CSV** yang berisi daftar node beserta peran (`Role`), probabilitas rata-rata (`AvgProb`), dan rasio inbound (`InRatio`).                |
