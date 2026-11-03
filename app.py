from pathlib import Path
import re
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

DATA_FILE = Path(__file__).parent / "players_data-2026_2027.csv"
MIN_ANALYSIS_MINUTES = 90
DEFAULT_RECOMMENDATION_MINUTES = 270

OUTFIELD_BASE = ["Gls", "Ast", "Sh", "SoT", "Crs", "TklW", "Int"]
OUTFIELD_FEATURES = [f"{col}_per90" for col in OUTFIELD_BASE]
GOALKEEPER_FEATURES = ["GA90", "Save%", "CS%"]
FEATURES_BY_ROLE = {
    "FW": OUTFIELD_FEATURES,
    "MF": OUTFIELD_FEATURES,
    "DF": OUTFIELD_FEATURES,
    "GK": GOALKEEPER_FEATURES,
}
K_BY_ROLE = {"FW": 4, "MF": 2, "DF": 2, "GK": 3}

st.set_page_config(
    page_title="Football Player Scout",
    page_icon="⚽",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1180px;}
      .hero {padding: 1.5rem 1.7rem; border-radius: 18px;
             background: linear-gradient(120deg,#10251e,#132c3a); margin-bottom: 1rem;}
      .hero h1 {margin: 0; color: #f4fff8;}
      .hero p {margin: .45rem 0 0; color: #b9d4ca;}
      .rec-card {padding: 1rem 1.1rem; border: 1px solid #263d48;
                 border-radius: 14px; margin: .55rem 0; background: #13232c;}
      .rec-name {font-size: 1.1rem; font-weight: 700;}
      .rec-meta {color: #b3c1c9; font-size: .9rem; margin-top: .25rem;}
      .score {color: #58e6a3; font-weight: 700; font-size: 1.05rem;}
      .note {color: #a9bac2; font-size: .85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Membaca dan menyiapkan statistik pemain...")
def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"File data tidak ditemukan: {DATA_FILE.name}. "
            "Letakkan CSV utama di folder yang sama dengan app.py."
        )

    df = pd.read_csv(DATA_FILE)
    required = ["Player", "Pos", "Squad", "Comp", "Age", "MP", "Min"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan di CSV: {missing}")

    numeric_cols = set(OUTFIELD_BASE + ["GA90", "Save%", "CS%"])
    numeric_cols.update(["Min", "Age", "MP"])
    for col in numeric_cols.intersection(df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Role"] = df["Pos"].fillna("").astype(str).str.split(",").str[0].str.strip()
    df = df.loc[df["Min"] >= MIN_ANALYSIS_MINUTES].copy()
    denominator = df["Min"].replace(0, np.nan) / 90
    for col in OUTFIELD_BASE:
        if col in df.columns:
            df[f"{col}_per90"] = df[col] / denominator

    df["Cluster"] = -1
    for role, features in FEATURES_BY_ROLE.items():
        role_df = df.loc[df["Role"] == role].dropna(subset=features)
        if len(role_df) < 2:
            continue
        X = StandardScaler().fit_transform(role_df[features])
        n_clusters = min(K_BY_ROLE[role], len(role_df) - 1)
        if n_clusters >= 2:
            labels = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
            ).fit_predict(X)
            df.loc[role_df.index, "Cluster"] = labels

    return df


def normalize_title(value):
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", value.lower())


@st.cache_data(ttl=86400, max_entries=1024, show_spinner=False)
def get_player_photo(player_name, squad):
    """Try a football-player image API first, then a matching Wikipedia thumbnail."""
    wanted = normalize_title(player_name)
    squad_key = normalize_title(squad)
    headers = {"User-Agent": "FootballPlayerScout/1.0 (educational project)"}

    # TheSportsDB has player-specific cutout and thumbnail fields.
    try:
        response = requests.get(
            f"https://www.thesportsdb.com/api/v1/json/3/searchplayers.php",
            params={"p": player_name},
            headers=headers,
            timeout=8,
        )
        response.raise_for_status()
        records = response.json().get("player") or []
        matches = []
        for record in records:
            result_name = normalize_title(record.get("strPlayer", ""))
            sport = str(record.get("strSport", "")).lower()
            if result_name != wanted or sport not in {"soccer", "football"}:
                continue
            result_team = normalize_title(record.get("strTeam", ""))
            team_match = bool(
                squad_key and result_team and
                (squad_key in result_team or result_team in squad_key)
            )
            image = record.get("strCutout") or record.get("strThumb")
            if image:
                matches.append((0 if team_match else 1, image))
        if matches:
            matches.sort(key=lambda item: item[0])
            return matches[0][1]
    except (requests.RequestException, ValueError):
        pass

    # Fallback: search Wikipedia and accept an exact, extended, or token-complete title.
    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f'"{player_name}" footballer {squad}',
        "gsrnamespace": 0,
        "gsrlimit": 8,
        "prop": "pageimages|pageterms",
        "piprop": "thumbnail",
        "pithumbsize": 400,
        "wbptterms": "description",
    }
    tokens = [normalize_title(token) for token in str(player_name).split() if len(token) > 2]

    try:
        response = requests.get(api_url, params=params, headers=headers, timeout=8)
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
    except (requests.RequestException, ValueError):
        return None

    candidates = []
    for page in pages.values():
        title_key = normalize_title(page.get("title", ""))
        if title_key == wanted:
            match_rank = 0
        elif title_key.startswith(wanted) and len(title_key) - len(wanted) <= 40:
            match_rank = 1
        elif tokens and all(token in title_key for token in tokens):
            match_rank = 2
        else:
            continue
        thumbnail = page.get("thumbnail", {}).get("source")
        if thumbnail:
            candidates.append((match_rank, page.get("index", 999), thumbnail))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


try:
    players = load_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.markdown(
    """
    <div class="hero">
      <h1>⚽ Football Player Scout</h1>
      <p>Temukan pemain dengan profil statistik yang mirip, dalam peran bermain yang sama.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Atur pencarian")
roles_available = [r for r in ["FW", "MF", "DF", "GK"] if (players["Role"] == r).any()]
role_filter_options = ["Semua peran"] + roles_available
role_filter = st.sidebar.selectbox("Filter posisi (opsional)", role_filter_options, index=0)
min_minutes = st.sidebar.slider(
    "Minimal menit kandidat",
    min_value=90,
    max_value=900,
    value=DEFAULT_RECOMMENDATION_MINUTES,
    step=90,
    help="Ambang ini hanya menyaring pemain kandidat yang direkomendasikan.",
)
top_n = st.sidebar.slider("Jumlah rekomendasi", 3, 10, 5)
search_text = st.sidebar.text_input(
    "Cari nama pemain",
    placeholder="Ketik sebagian nama, misalnya haaland",
).strip()

# Hanya tampilkan pemain yang fitur statistiknya cukup lengkap untuk perannya.
valid_indices = []
for available_role in roles_available:
    role_features_for_filter = FEATURES_BY_ROLE[available_role]
    valid_indices.extend(
        players.loc[players["Role"] == available_role]
        .dropna(subset=role_features_for_filter)
        .index.tolist()
    )
search_df = players.loc[players.index.isin(valid_indices)].copy()
if role_filter != "Semua peran":
    search_df = search_df.loc[search_df["Role"] == role_filter]
if search_text:
    search_mask = search_df["Player"].astype(str).str.contains(
        re.escape(search_text), case=False, na=False
    )
    search_df = search_df.loc[search_mask]

if search_df.empty:
    st.sidebar.warning("Nama pemain tidak ditemukan. Coba kata lain atau pilih Semua peran.")
    st.stop()

# Peran rekomendasi ditentukan otomatis dari pemain yang dipilih.
selected_index = st.sidebar.selectbox(
    "Pilih pemain",
    options=search_df.index.tolist(),
    format_func=lambda index: (
        f"{search_df.loc[index, 'Player']} — {search_df.loc[index, 'Pos']} · "
        f"{search_df.loc[index, 'Squad']} ({search_df.loc[index, 'Comp']})"
    ),
)
selected = players.loc[selected_index]
role = selected["Role"]
features = FEATURES_BY_ROLE[role]
role_df = players.loc[players["Role"] == role].dropna(subset=features).copy()
selected = role_df.loc[selected_index]


def player_label(index):
    row = search_df.loc[index]
    return f"{row['Player']} — {row['Squad']} ({row['Comp']})"

player_indices = search_df.index.tolist()
selected_index = st.sidebar.selectbox(
    "Pilih pemain",
    options=player_indices,
    format_func=player_label,
)
selected = role_df.loc[selected_index]

# Skalakan dalam kelompok peran yang sama, seperti pada eksperimen notebook.
scaler = StandardScaler()
X_role = scaler.fit_transform(role_df[features])
selected_position = role_df.index.get_loc(selected_index)
query_vector = X_role[[selected_position]]

candidate_mask = (
    (role_df["Min"] >= min_minutes)
    & (role_df.index != selected_index)
)
candidate_positions = np.flatnonzero(candidate_mask.to_numpy())

st.subheader("Pemain pilihan")
profile_cols = st.columns([1, 3, 1, 1, 1])
selected_photo = get_player_photo(str(selected["Player"]), str(selected["Squad"]))
with profile_cols[0]:
    if selected_photo:
        st.image(selected_photo, width=112)
    else:
        initials = "".join(part[0] for part in str(selected["Player"]).split() if part)[:2].upper()
        st.markdown(
            f"<div style='width:96px;height:96px;border-radius:18px;background:#173e31;"
            f"color:#78e6ac;display:flex;align-items:center;justify-content:center;"
            f"font-size:30px;font-weight:700'>{initials or '⚽'}</div>",
            unsafe_allow_html=True,
        )
with profile_cols[1]:
    st.markdown(f"### {selected['Player']}")
    st.caption(f"{selected['Pos']} · {selected['Squad']} · {selected['Comp']}")
with profile_cols[2]:
    st.metric("Menit bermain", f"{int(selected['Min']):,}")
with profile_cols[3]:
    st.metric("Pertandingan", f"{int(selected['MP'])}")
with profile_cols[4]:
    st.metric("Usia", "—" if pd.isna(selected["Age"]) else f"{selected['Age']:.0f}")

if len(candidate_positions) == 0:
    st.info("Tidak ada kandidat yang memenuhi batas menit. Coba turunkan nilai minimal menit.")
    st.stop()

candidate_matrix = X_role[candidate_positions]
scores = cosine_similarity(query_vector, candidate_matrix)[0]
recommendations = role_df.iloc[candidate_positions].copy()
recommendations["Similarity"] = scores
recommendations = recommendations.sort_values("Similarity", ascending=False).head(top_n)

st.subheader("Pemain dengan statistik serupa")
for rank, (_, player) in enumerate(recommendations.iterrows(), start=1):
    photo_col, details_col, score_col = st.columns([1, 5, 1])
    photo = get_player_photo(str(player["Player"]), str(player["Squad"]))
    with photo_col:
        if photo:
            st.image(photo, width=82)
        else:
            initials = "".join(part[0] for part in str(player["Player"]).split() if part)[:2].upper()
            st.markdown(
                f"<div style='width:68px;height:68px;border-radius:14px;background:#173e31;"
                f"color:#78e6ac;display:flex;align-items:center;justify-content:center;"
                f"font-size:23px;font-weight:700'>{initials or '⚽'}</div>",
                unsafe_allow_html=True,
            )
    with details_col:
        st.markdown(
            f"""
            <div class="rec-card">
              <div class="rec-name">{rank}. {player['Player']}</div>
              <div class="rec-meta">{player['Pos']} · {player['Squad']} · {player['Comp']}</div>
              <div class="rec-meta">Menit bermain: {int(player['Min']):,} · Cluster: {int(player['Cluster'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with score_col:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="score">Cosine<br>{player["Similarity"]:.3f}</div>',
            unsafe_allow_html=True,
        )

st.caption("Skor Cosine adalah nilai kemiripan statistik, bukan persentase atau peluang.")

st.subheader("Perbandingan profil")
plot_recommendations = recommendations.head(3)
percentiles = role_df[features].rank(pct=True)
if role == "GK" and "GA90" in features:
    # Kebobolan lebih sedikit berarti performa lebih baik.
    percentiles["GA90"] = 1 - percentiles["GA90"]

labels_by_feature = {
    "Gls_per90": "Gol/90",
    "Ast_per90": "Assist/90",
    "Sh_per90": "Tembakan/90",
    "SoT_per90": "Tepat sasaran/90",
    "Crs_per90": "Umpan silang/90",
    "TklW_per90": "Tekel/90",
    "Int_per90": "Intersepsi/90",
    "GA90": "Kebobolan/90*",
    "Save%": "Save%",
    "CS%": "Clean sheet%",
}
labels = [labels_by_feature.get(col, col) for col in features]
angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
angles += angles[:1]
fig, ax = plt.subplots(figsize=(7.2, 6.2), subplot_kw={"polar": True})

for index, row in [(selected_index, selected)] + list(plot_recommendations.iterrows()):
    values = percentiles.loc[index, features].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=str(row["Player"]))
    ax.fill(angles, values, alpha=0.07)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["25", "50", "75", "100"])
ax.set_title(f"Persentil dibanding pemain {role} lain", pad=22)
ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.1))
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)
st.caption("Radar chart menunjukkan persentil di antara pemain dengan peran yang sama. Untuk kiper, GA90 dibalik agar nilai lebih tinggi berarti lebih sedikit kebobolan.")

with st.expander("Lihat tabel statistik lengkap"):
    display_features = ["Player", "Pos", "Squad", "Min", "Similarity"] + features
    selected_table = role_df.loc[[selected_index], ["Player", "Pos", "Squad", "Min"] + features].copy()
    selected_table.insert(4, "Similarity", np.nan)
    rec_table = recommendations[display_features].copy()
    st.dataframe(
        pd.concat([selected_table, rec_table], ignore_index=True).round(3),
        use_container_width=True,
        hide_index=True,
    )

with st.expander("Tentang metode"):
    st.write(
        "Pemain dibandingkan hanya dengan peran utama yang sama. Statistik pemain lapangan dihitung per 90 menit, "
        "fitur dinormalisasi menggunakan StandardScaler, lalu peringkat kemiripan dihitung dengan Cosine Similarity. "
        "K-Means membentuk cluster pemain per peran; label cluster ditampilkan sebagai konteks tambahan."
    )
    st.write(f"Data yang memenuhi syarat analisis (minimal {MIN_ANALYSIS_MINUTES} menit): {len(players):,} pemain.")
