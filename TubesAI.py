# ============================================================
#  DECISION TREE - GRADING KOPI ARABIKA BERDASARKAN SCA GACCS
#  Dataset: Coffee Quality Institute (CQI) - 207 sampel
# ============================================================

# ── IMPORT LIBRARY ──────────────────────────────────────────
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder

# pandas   → manipulasi data (tabel/dataframe)
# numpy    → operasi numerik
# matplotlib & seaborn → visualisasi grafik
# sklearn  → library machine learning utama:
#   - DecisionTreeClassifier : algoritma Decision Tree nya
#   - train_test_split       : bagi data train & test
#   - classification_report  : laporan akurasi per kelas
#   - confusion_matrix       : matriks kesalahan prediksi


# ── STEP 1: LOAD DATA ───────────────────────────────────────
print("=" * 60)
print("STEP 1: LOAD DATA")
print("=" * 60)

df = pd.read_excel('df_arabica_clean.xlsx', sheet_name='Arabica Coffee Data')

print(f"Jumlah baris (sampel kopi) : {len(df)}")
print(f"Jumlah kolom               : {len(df.columns)}")

# Kolom yang akan dipakai — 15 fitur input + 1 target (dibuat di step 2)
FITUR = [
    # 10 skor cupping → penilaian rasa oleh Q-Grader
    'Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body',
    'Balance', 'Uniformity', 'Clean Cup', 'Sweetness', 'Overall',
    # Total skor → penjumlahan semua komponen cupping di atas
    'Total Cup Points',
    # 5 atribut cacat fisik → penentu lolos/tidak standar SCA
    'Category One Defects', # cacat primer (biji hitam, berlubang serangga)
    'Category Two Defects', # cacat sekunder (kulit ari, biji pecah)
    'Quakers',              # biji yang tidak matang sempurna
    'Moisture Percentage',  # kadar air biji
    'Color'                 # warna biji
]

print(f"\nFitur yang dipakai ({len(FITUR)} kolom):")
for f in FITUR:
    print(f"  - {f}")

# ── STEP 2: BUAT LABEL GRADE SCA ────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: BUAT LABEL GRADE BERDASARKAN STANDAR SCA GACCS")
print("=" * 60)

# Dataset CQI tidak punya kolom grade → kita buat manual
# Aturan SCA GACCS (SCA Defect Handbook 2004 + Cupping Protocol):
#
#   Syarat LOLOS (harus semua terpenuhi):
#     1. Category One Defects = 0   (nol toleransi cacat primer)
#     2. Category Two Defects ≤ 5   (maks 5 cacat sekunder)
#     3. Quakers = 0                (nol toleransi biji mentah)
#     4. Color > Greenish           (blue-Green, bluish-Green, green)
#
#   Setelah lolos syarat defect, cek skor:
#     Total Cup Points ≥ 85  → Excellent   (SCA: "Outstanding/Excellent")
#     Total Cup Points ≥ 80  → Very Good   (SCA: "Very Good/Specialty")
#     Total Cup Points < 80  → Not Classified

def buat_label_sca(row):
    """
    Fungsi ini diterapkan ke setiap baris (sampel kopi).
    row['kolom'] → mengakses nilai kolom tertentu dari baris itu.
    """
    # Cek syarat cacat — kalau gagal, langsung Not Classified
    # tidak perlu cek skor sama sekali
    if row['Category One Defects'] > 0:
        return 'Not Classified'
    if row['Category Two Defects'] > 5:
        return 'Not Classified'
    if row['Quakers'] > 0:
        return 'Not Classified'
    if row['Color'] != 'blue-Green' and row['Color'] != 'bluish-Green' and row['Color'] != 'green':
        return 'Not Classified'

    # Lolos syarat cacat → cek skor cupping
    if row['Total Cup Points'] >= 90:
        return 'Outstanding'
    if row['Total Cup Points'] >= 85:
        return 'Excellent'
    elif row['Total Cup Points'] >= 80:
        return 'Very Good'
    else:
        return 'Not Classified'

# Terapkan fungsi ke seluruh dataset → hasilkan kolom baru 'grade_sca'
# axis=1 → terapkan per baris (bukan per kolom)
df['grade_sca'] = df.apply(buat_label_sca, axis=1)

print("\nDistribusi label grade SCA di seluruh dataset:")
distribusi = df['grade_sca'].value_counts()
for grade, jumlah in distribusi.items():
    persen = jumlah / len(df) * 100
    print(f"  {grade:20s}: {jumlah} sampel ({persen:.1f}%)")

# Cek sampel Premium (TCP 75-79.99) — terlalu sedikit untuk dilatih
premium = df[df['Total Cup Points'].between(75, 79.99)]
print(f"\nSampel dengan TCP 75-79.99 (Premium): {len(premium)} sampel")
print("→ Terlalu sedikit, sudah masuk 'Not Classified' di atas")


# ── STEP 3: PERSIAPAN DATA X DAN y ──────────────────────────
print("\n" + "=" * 60)
print("STEP 3: PERSIAPAN FITUR INPUT (X) DAN TARGET (y)")
print("=" * 60)

# convert warna biji menjadi angka untuk model
order = [
    "blue-green", "bluish-green", "green", "greenish",
    "yellow-green", "pale-yellow", "yellowish", "brownish"
]
# menggunakan len(order) - 1, blue-green = 8 sedangkan brownish = 1
mapping = {name.strip().lower(): len(order) - i for i, name in enumerate(order)}

def encode_color_series(s):
    return s.astype(str).str.strip().str.lower().map(mapping).fillna(0).astype(int)

# X = semua fitur input (15 kolom) → yang "dilihat" model untuk belajar
X = df[FITUR]
X['Color'] = encode_color_series(X['Color'])

# y = label grade yang baru dibuat → yang mau diprediksi model
y = df['grade_sca']

print(f"Ukuran X (fitur input) : {X.shape}  → {X.shape[0]} sampel, {X.shape[1]} kolom")
print(f"Ukuran y (target)      : {y.shape}  → {y.shape[0]} label")

print("\nContoh 5 baris pertama X dan y berdampingan:")
contoh = X.head(5).copy()
contoh['>> GRADE'] = y.head(5).values
print(contoh[['Total Cup Points', 'Category One Defects',
              'Category Two Defects', 'Quakers', 'Color', '>> GRADE']].to_string())
# bisa lihat langsung hubungan antara nilai kolom dan grade yang terbentuk


# ── STEP 4: SPLIT DATA TRAIN & TEST ─────────────────────────
print("\n" + "=" * 60)
print("STEP 4: SPLIT DATA TRAIN (80%) DAN TEST (20%)")
print("=" * 60)

# train_test_split → fungsi sklearn untuk bagi data secara acak
# test_size=0.2    → 20% untuk test, 80% untuk train
# random_state=42  → angka seed supaya hasil split selalu sama tiap dijalankan
#                    (42 konvensi umum, bisa diganti angka lain)
# stratify=y       → pastikan proporsi tiap kelas seimbang di train & test
#                    penting karena kelas tidak seimbang jumlahnya
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Total sampel          : {len(X)}")
print(f"Sampel untuk TRAINING : {len(X_train)} ({len(X_train)/len(X)*100:.0f}%)")
print(f"Sampel untuk TESTING  : {len(X_test)} ({len(X_test)/len(X)*100:.0f}%)")

print("\nDistribusi kelas di Training set:")
for grade, jml in y_train.value_counts().items():
    print(f"  {grade:20s}: {jml} sampel")

print("\nDistribusi kelas di Testing set:")
for grade, jml in y_test.value_counts().items():
    print(f"  {grade:20s}: {jml} sampel")
# Cek: proporsinya harus mirip antara train dan test (hasil stratify)


# ── STEP 5: TRAINING MODEL DECISION TREE ────────────────────
print("\n" + "=" * 60)
print("STEP 5: TRAINING MODEL DECISION TREE")
print("=" * 60)

# DecisionTreeClassifier → ini algoritmanya
# criterion='gini'  → pakai Gini Impurity untuk cari split terbaik
#                     Gini mengukur seberapa "kotor" sebuah node
#                     (0 = semua isi satu kelas = murni)
# max_depth=4       → batas kedalaman pohon
#                     tanpa batas → pohon terlalu kompleks → overfitting
#                     (hafal data latih tapi gagal di data baru)
# min_samples_split=5 → minimal 5 sampel di node untuk bisa di-split lagi
#                       mencegah pohon terlalu detail ke data noise
# random_state=42   → supaya hasil sama tiap dijalankan
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    min_samples_split=5,
    random_state=42
)

# .fit() → ini proses "belajar" nya
# model membaca X_train dan y_train, lalu mencari aturan
# (pertanyaan IF-THEN) yang paling baik memisahkan tiap kelas
model.fit(X_train, y_train) ##

print("Model berhasil dilatih!")
print(f"\nKedalaman pohon yang terbentuk : {model.get_depth()} level")
print(f"Jumlah leaf node (keputusan akhir): {model.get_n_leaves()}")

# Feature importance → seberapa besar kontribusi tiap fitur dalam keputusan
print("\nFeature Importance (kontribusi tiap fitur):")
importance_df = pd.DataFrame({
    'Fitur': FITUR,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in importance_df.iterrows():
    bar = '█' * int(row['Importance'] * 50)
    print(f"  {row['Fitur']:25s}: {row['Importance']:.4f} {bar}")


# ── STEP 6: PREDIKSI & EVALUASI ─────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: PREDIKSI DAN EVALUASI MODEL")
print("=" * 60)

# .predict() → model menerapkan aturan yang sudah dipelajari
# ke data test yang belum pernah dilihat sebelumnya
y_pred = model.predict(X_test)

# Akurasi keseluruhan
akurasi = accuracy_score(y_test, y_pred)
print(f"\nAkurasi keseluruhan: {akurasi:.4f} ({akurasi*100:.2f}%)")
print("→ Artinya dari setiap 100 sampel test, model benar sebanyak itu")

# Classification report → detail per kelas
print("\nLaporan per kelas:")
print(classification_report(y_test, y_pred))
# Precision : dari semua yang diprediksi kelas X, berapa % yang benar?
# Recall    : dari semua yang sebenarnya kelas X, berapa % yang tertangkap?
# F1-score  : rata-rata harmonis precision & recall (0-1, makin tinggi makin baik)
# Support   : jumlah sampel asli per kelas di test set


# # ── STEP 7: VISUALISASI ─────────────────────────────────────
# print("\n" + "=" * 60)
# print("STEP 7: MEMBUAT VISUALISASI")
# print("=" * 60)

# fig = plt.figure(figsize=(28, 20))
# fig.patch.set_facecolor('#FAFAFA')

# # ── 7A: Pohon Keputusan ──────────────────────────────────────
# ax1 = fig.add_subplot(2, 2, (1, 2))  # baris atas, span 2 kolom
# plot_tree(
#     model,
#     feature_names=FITUR,
#     class_names=model.classes_,
#     filled=True,          # warnai node berdasarkan kelas mayoritas
#     rounded=True,         # sudut node membulat
#     fontsize=8,
#     ax=ax1,
#     impurity=True,        # tampilkan nilai Gini
#     proportion=False
# )
# ax1.set_title('Pohon Keputusan — Grading Kopi Arabika (SCA GACCS)',
#               fontsize=14, fontweight='bold', pad=12)

# # ── 7B: Confusion Matrix ─────────────────────────────────────
# ax2 = fig.add_subplot(2, 2, 3)
# kelas = model.classes_
# cm = confusion_matrix(y_test, y_pred, labels=kelas)

# sns.heatmap(
#     cm, annot=True, fmt='d', cmap='YlOrBr',
#     xticklabels=kelas, yticklabels=kelas,
#     linewidths=0.5, linecolor='white',
#     ax=ax2, cbar=False, annot_kws={'size': 13}
# )
# ax2.set_xlabel('Prediksi Model', fontsize=11)
# ax2.set_ylabel('Label Asli', fontsize=11)
# ax2.set_title('Confusion Matrix\n(diagonal = prediksi benar)', fontsize=12, fontweight='bold')
# # Cara baca: baris = kelas asli, kolom = prediksi
# # Angka di diagonal = prediksi BENAR
# # Angka di luar diagonal = prediksi SALAH (misclassified)

# # ── 7C: Feature Importance ───────────────────────────────────
# ax3 = fig.add_subplot(2, 2, 4)
# imp_sorted = importance_df[importance_df['Importance'] > 0]
# colors = ['#2E7D32' if imp > 0.1 else '#81C784' for imp in imp_sorted['Importance']]
# bars = ax3.barh(imp_sorted['Fitur'], imp_sorted['Importance'], color=colors, edgecolor='white')

# # Tambahkan label nilai di tiap bar
# for bar, val in zip(bars, imp_sorted['Importance']):
#     ax3.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
#              f'{val:.4f}', va='center', fontsize=9)

# ax3.set_xlabel('Importance Score', fontsize=11)
# ax3.set_title('Feature Importance\n(fitur paling berpengaruh di atas)',
#               fontsize=12, fontweight='bold')
# ax3.set_xlim(0, imp_sorted['Importance'].max() * 1.2)
# ax3.invert_yaxis()  # fitur terpenting di atas
# ax3.axvline(x=0.1, color='red', linestyle='--', alpha=0.4, label='threshold 0.1')
# ax3.legend(fontsize=9)

# plt.tight_layout(pad=3)
# plt.savefig('/mnt/user-data/outputs/decision_tree_kopi.png',
#             dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
# plt.close()
# print("Visualisasi disimpan!")

# ── STEP 8: ATURAN IF-THEN DARI POHON ───────────────────────
print("\n" + "=" * 60)
print("STEP 8: ATURAN IF-THEN YANG DIPELAJARI MODEL")
print("=" * 60)
print("(Ini yang jadi bahan pembahasan di paper lo)\n")
rules = export_text(model, feature_names=FITUR)
print(rules)
# |--- berarti percabangan (node)
# leaf berarti keputusan akhir (grade)
# Baca dari atas ke bawah mengikuti kondisi yang terpenuhi

print("\n" + "=" * 60)
print("SELESAI!")
print("=" * 60)

