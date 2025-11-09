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
