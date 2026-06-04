import streamlit as st
import pandas as pd

from utils.inference import predict_single

st.set_page_config(
    page_title="Analisis Sentimen dan Aspek",
    page_icon="📊",
    layout="wide"
)

# HEADER

st.markdown("""
    <h2 style='text-align: center; color: #1F3C88;'>
        ANALISIS SENTIMEN BERBASIS ASPEK MENGGUNAKAN BERT DAN PELABELAN  POWERSET TERHADAP ULASAN LAYANAN RUMAH SAKIT
    </h2>
""", unsafe_allow_html=True)

# st.markdown("""
#     <h3 style='text-align: center; color:gray;'>
#         Sistem analisis sentimen berbasis IndoBERT untuk mendeteksi aspek dan polaritas sentimen pada ulasan pengguna.
#     </h3>
# """, unsafe_allow_html=True)

st.divider()

# INPUT

st.subheader("Masukkan Ulasan:")

text = st.text_area(
    "",
    placeholder="Contoh: Pelayanan dokter sangat ramah.",
    height=180
)

# BUTTON

if st.button("Prediksi Sentimen", use_container_width=True):

    if not text.strip():

        st.warning("Masukkan ulasan terlebih dahulu.")

    else:

        with st.spinner("Sedang memproses prediksi..."):

            result = predict_single(text)

        st.success("Prediksi berhasil dilakukan.")

        
        # PREPROCESSING
        

        st.divider()

        st.subheader("Ringkasan Hasil")

        aspect_labels = {
            "kpms_pos": "Kualitas Pelayanan Medis dan Staf (Positif)",
            "kpms_neg": "Kualitas Pelayanan Medis dan Staf (Negatif)",

            "fi_pos": "Fasilitas dan Infrastruktur (Positif)",
            "fi_neg": "Fasilitas dan Infrastruktur (Negatif)",

            "wt_pos": "Waktu Tunggu (Positif)",
            "wt_neg": "Waktu Tunggu (Negatif)",

            "bl_pos": "Biaya Layanan (Positif)",
            "bl_neg": "Biaya Layanan (Negatif)"
        }

        detected = [
            aspect_labels[k]
            for k, v in result["aspects"].items()
            if v == 1
        ]

        if detected:
            st.info(
                "Sentimen yang terdeteksi: "
                + ", ".join(detected)
            )
        else:
            st.warning(
                "Tidak ditemukan aspek sentimen yang terdeteksi."
            )


        # =========================
        # ASPEK DAN SENTIMEN
        # =========================

        st.divider()

        st.subheader("Deteksi Aspek dan Sentimen")

        rows = []

        for key, value in result["aspects"].items():

            status = "Terdeteksi" if value == 1 else "Tidak Terdeteksi"

            rows.append({
                "Aspek dan Sentimen": aspect_labels[key],
                "Status": status
            })

        df_aspek = pd.DataFrame(rows)

        def highlight_status(val):
            if val == "Terdeteksi":
                return "background-color: #D4EDDA; color: #155724; font-weight: bold;"
            else:
                return "background-color: #F8D7DA; color: #721C24;"

        styled_df = df_aspek.style.map(
            highlight_status,
            subset=["Status"]
        )

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )


        # =========================
        # HASIL PREDIKSI
        # =========================

        st.divider()

        st.subheader("Informasi Prediksi")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Label Class",
                value=result["class"]
            )

        with col2:
            st.metric(
                label="Kombinasi Biner",
                value=result["binary"]
            )


        # =========================
        # DETAIL PREPROCESSING
        # =========================

        st.divider()

        with st.expander("Detail Preprocessing", expanded=False):

            prep = result["preprocessing"]

            df_prep = pd.DataFrame({
                "Tahapan": [
                    "Cleaning",
                    "Case Folding",
                    "Tokenisasi",
                    "Normalisasi",
                    "Stopword Removal",
                    "Stemming",
                    "Final Text"
                ],
                "Hasil": [
                    prep["cleaning"],
                    prep["case_folding"],
                    ", ".join(prep["tokenisasi"]),
                    ", ".join(prep["normalisasi"]),
                    ", ".join(prep["stopword_removal"]),
                    ", ".join(prep["stemming"]),
                    prep["final_text"]
                ]
            })

            st.dataframe(
                df_prep,
                use_container_width=True,
                hide_index=True
            )