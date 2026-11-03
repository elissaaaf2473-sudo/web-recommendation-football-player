import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output

hasil_radar = widgets.Output()
display(hasil_radar)

def tampilkan_radar(_=None):
    with hasil_radar:
        clear_output(wait=True)

        nama = pilihan_pemain.value
        pilihan = df_eligible[df_eligible['Player'] == nama].iloc[0]
        role = pilihan['Role']
        fitur = role_features[role]
        rekomendasi = rekomendasikan_pemain(
            nama,
            top_n=3,
            min_menit=batas_menit.value
        )

        if rekomendasi is None or rekomendasi.empty:
            print("Tidak ada rekomendasi untuk batas menit tersebut.")
            return

        # Persentil dihitung di antara pemain dengan peran yang sama.
        role_df = df_eligible[df_eligible['Role'] == role]
        percentile = role_df[fitur].rank(pct=True)

        # Untuk kiper, GA90 yang lebih rendah berarti lebih baik.
        if role == 'GK' and 'GA90' in fitur:
            percentile['GA90'] = 1 - percentile['GA90']

        nama_fitur = {
            'Gls_per90': 'Gol/90',
            'Ast_per90': 'Assist/90',
            'Sh_per90': 'Tembakan/90',
            'SoT_per90': 'Tepat sasaran/90',
            'Crs_per90': 'Umpan silang/90',
            'TklW_per90': 'Tekel/90',
            'Int_per90': 'Intersepsi/90',
            'GA90': 'Kebobolan/90*',
            'Save%': 'Save%',
            'CS%': 'Clean sheet%'
        }

        labels = [nama_fitur.get(f, f) for f in fitur]
        sudut = np.linspace(0, 2 * np.pi, len(fitur), endpoint=False).tolist()
        sudut += sudut[:1]

        fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={'polar': True})

        # Garis pemain pilihan
        idx_pilihan = pilihan.name
        nilai = percentile.loc[idx_pilihan, fitur].tolist()
        nilai += nilai[:1]
        ax.plot(sudut, nilai, linewidth=2.5, label=f"{nama} (pilihan)")
        ax.fill(sudut, nilai, alpha=0.12)

        # Garis tiga rekomendasi teratas
        for idx, (_, pemain) in enumerate(rekomendasi.iterrows(), start=1):
            nilai = percentile.loc[pemain.name, fitur].tolist()
            nilai += nilai[:1]
            ax.plot(sudut, nilai, linewidth=1.5, label=pemain['Player'])
            ax.fill(sudut, nilai, alpha=0.04)

        ax.set_xticks(sudut[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.50, 0.75, 1.0])
        ax.set_yticklabels(['25', '50', '75', '100'])
        ax.set_title(
            f"Perbandingan profil {role} — persentil dalam peran yang sama",
            pad=24
        )
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.1))
        fig.tight_layout()

        display(fig)
        plt.close(fig)

# Tombol yang sama akan memperbarui aplikasi dan radar chart.
tombol_cari.on_click(tampilkan_radar)
tampilkan_radar()