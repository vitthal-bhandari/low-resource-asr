"""
Language Map — Mozilla Common Voice Spontaneous Speech Shared Task
21 low-resource languages plotted on a publication-quality world map.

Requirements:
    pip install cartopy matplotlib numpy

Output:
    language_map.pdf  (vector, for LaTeX inclusion)
    language_map.png  (300 dpi raster, for quick preview)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ── 1. Language data ──────────────────────────────────────────────────────────
LANGUAGES = [
    # Africa
    ("bxk", "Bukusu",              34.55,   0.60,  "Africa"),
    ("cgg", "Chiga",               30.00,  -1.30,  "Africa"),
    ("kcn", "Nubi",                32.50,   3.80,  "Africa"),
    ("koo", "Konzo",               29.90,   0.10,  "Africa"),
    ("led", "Lendu",               30.20,   1.90,  "Africa"),
    ("lke", "Kenyi",               31.60,   3.60,  "Africa"),
    ("lth", "Thur",                31.80,   6.20,  "Africa"),
    ("ruc", "Ruuli",               31.90,   1.50,  "Africa"),
    ("rwm", "Amba",                30.00,   1.00,  "Africa"),
    ("ttj", "Rutoro",              30.50,   0.70,  "Africa"),
    ("ukv", "Kuku",                32.10,   4.10,  "Africa"),
    # Americas
    ("hch", "Wixárika",           -104.00,  22.00, "Americas"),
    ("meh", "Southwestern\nTlaxiaco Mixtec", -97.70,  17.10, "Americas"),
    ("mmc", "Michoacán\nMazahua",  -100.20,  19.60, "Americas"),
    ("tob", "Toba Qom",            -60.70, -24.80, "Americas"),
    ("top", "Papantla\nTotonac",    -97.30,  20.40, "Americas"),
    # Asia / Pacific
    ("bew", "Betawi",              106.85,  -6.20, "Asia/Pacific"),
    ("pne", "Western\nPenan",      114.50,   3.60, "Asia/Pacific"),
    # Europe
    ("aln", "Gheg\nAlbanian",       20.00,  41.50, "Europe"),
    ("el-CY", "Cypriot\nGreek",     33.00,  35.10, "Europe"),
    ("sco", "Scots",                -4.00,  56.50, "Europe"),
]

# ── 2. Colours ────────────────────────────────────────────────────────────────
# Derived from LaTeX table tints in table1-3.tex:
#   tintAfrica  = #FFF5E6 → #C47A00  mid warm amber    (lightened from #8B5E00)
#   tintAmericas= #F0FFF0 → #1A6B1A  dark forest green (unchanged)
#   tintAsia    = #F0F4FF → #1A3D8B  dark steel blue   (unchanged)
#   tintEurope  = #FDF0FF → #9B59B6  mid lavender      (lightened from #6B1A8B)
CONTINENT_COLORS = {
    "Africa":       "#C47A00",
    "Americas":     "#1A6B1A",
    "Asia/Pacific": "#1A3D8B",
    "Europe":       "#9B59B6",
}

# ── 3. Crop extent ────────────────────────────────────────────────────────────
# Bottom trimmed: -38 + 10 = -28° (Toba Qom -24.8°S, label points UP → fits)
# Top restored:   52 + 10 = 62°N  (Scots at 56.5°N + downward label → fits)
LON_MIN, LON_MAX = -120, 130
LAT_MIN, LAT_MAX =  -28,  60

fig = plt.figure(figsize=(14, 6.5), dpi=150)
proj = ccrs.Mercator(central_longitude=15)
ax   = fig.add_subplot(1, 1, 1, projection=proj)
ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

# ── 4. Base map features ──────────────────────────────────────────────────────
ax.add_feature(cfeature.OCEAN,     facecolor="#D6EAF8", zorder=0)
ax.add_feature(cfeature.LAND,      facecolor="#F5F5EE", zorder=1)
ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#AAAAAA", zorder=2)
ax.add_feature(cfeature.BORDERS,   linewidth=0.25, edgecolor="#CCCCCC",
               linestyle=":", zorder=2)
ax.add_feature(cfeature.LAKES,     facecolor="#D6EAF8", zorder=2, alpha=0.6)
ax.add_feature(cfeature.RIVERS,    linewidth=0.3, edgecolor="#A8CFEA", zorder=2)

# ── 5. Label offsets (dx°, dy°) ──────────────────────────────────────────────
# Scots  (-8, -7): label SW/downward → fits within tighter top crop at 52°N
# Toba Qom (-10, +8): label NW/upward → fits within tighter bottom crop at -28°
LABEL_OFFSETS = {
    # Africa — evenly clocked around the dense Uganda/DRC cluster
    "lth": (  0,  14),   # Thur    → N
    "ukv": (  9,  12),   # Kuku    → NNE
    "kcn": ( 16,   7),   # Nubi    → ENE
    "lke": ( 15,   0),   # Kenyi   → E
    "ruc": ( 13,  -6),   # Ruuli   → ESE
    "bxk": (  9, -12),   # Bukusu  → SSE
    "cgg": (  0, -14),   # Chiga   → S
    "koo": ( -9, -10),   # Konzo   → SSW
    "rwm": (-16,  -2),   # Amba    → WSW
    "ttj": (-13,   6),   # Rutoro  → WNW
    "led": ( -8,  15),   # Lendu   → NNW
    # Americas
    "hch": (  0,  10),   # Wixárika  → N  (↑ upward)
    "mmc": (  -6, -13),   # Mazahua   → S  (↓ downward)
    "top": ( 16,   12),   # Totonac   → NE
    "meh": ( 28,  0),   # SW Mixtec → SE
    "tob": (-10,   8),   # Toba Qom  → NW (↑ upward, away from bottom edge)
    # Asia/Pacific
    "bew": ( 10,  -7),   # Betawi   → SE
    "pne": ( 7,   10),   # W. Penan → NE
    # Europe
    "aln":   (-9,   5),  # Gheg Albanian → NW
    "el-CY": ( 17,  -2),  # Cypriot Greek → SE
    "sco":   (-8,  -7),  # Scots → SW (↓ downward, away from top edge)
}

# ── 6. Fonts — 20% larger than previous (6.5 → 7.8pt, 5.5 → 6.6pt) ──────────
NAME_FONT = {"fontfamily": "DejaVu Sans", "fontsize": 10.9,
             "fontweight": "bold",   "fontstyle": "normal"}
ISO_FONT  = {"fontfamily": "DejaVu Sans", "fontsize": 6.6,
             "fontweight": "normal", "fontstyle": "italic"}

# ── 7. Plot pins + leader lines + labels ─────────────────────────────────────
GEODETIC = ccrs.Geodetic()

for iso, name, lon, lat, continent in LANGUAGES:
    color = CONTINENT_COLORS[continent]
    dx, dy = LABEL_OFFSETS.get(iso, (7, 4))
    lx, ly = lon + dx, lat + dy

    # Pin
    ax.plot(lon, lat,
            marker="o", markersize=5.5,
            color=color,
            markeredgecolor="white", markeredgewidth=1.0,
            transform=GEODETIC, zorder=5)

    # Leader line
    ax.annotate(
        "",
        xy=(lon, lat),
        xytext=(lx, ly),
        xycoords=GEODETIC._as_mpl_transform(ax),
        textcoords=GEODETIC._as_mpl_transform(ax),
        arrowprops=dict(arrowstyle="-", color=color, lw=0.75, alpha=0.9),
        zorder=4,
    )

    # Language name (bold)
    ax.text(
        lx, ly + 0.9,
        name,
        transform=GEODETIC,
        color=color,
        ha="center", va="bottom",
        linespacing=1.2,
        zorder=6,
        path_effects=[pe.withStroke(linewidth=2.8, foreground="white")],
        **NAME_FONT,
    )

    # ISO code (italic, smaller)
    ax.text(
        lx, ly - 0.9,
        f"({iso})",
        transform=GEODETIC,
        color=color,
        ha="center", va="top",
        zorder=6,
        path_effects=[pe.withStroke(linewidth=2.8, foreground="white")],
        **ISO_FONT,
    )

# ── 8. Legend — shifted up ~10% via bbox_to_anchor ───────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=CONTINENT_COLORS[c],
                   edgecolor="#CCCCCC", linewidth=1.2, label=c)
    for c in ["Africa", "Americas", "Asia/Pacific", "Europe"]
]
ax.legend(
    handles=legend_handles,
    title="Region", title_fontsize=12,
    loc="lower left",
    bbox_to_anchor=(0.01, 0.04),   # y=0.12 ≈ 10% up from axes bottom
    bbox_transform=ax.transAxes,
    framealpha=0.92, edgecolor="#CCCCCC",
    fancybox=False,
    handlelength=1.2, handleheight=1.0, borderpad=0.8,
    prop={"family": "DejaVu Sans", "size": 9.5},
)

# ── 9. Title & footnote ───────────────────────────────────────────────────────
# ax.set_title(
#     "Geographic Distribution of Languages in the Mozilla Common Voice Spontaneous Speech Dataset",
#     fontsize=11, fontweight="bold", pad=10, color="#222222",
#     fontfamily="DejaVu Sans",
# )
# fig.text(
#     0.99, 0.01,
#     "21 languages · 4 regions · Mozilla Common Voice",
#     ha="right", va="bottom", fontsize=6.5, color="#888888",
#     style="italic", fontfamily="DejaVu Sans",
# )

# ── 10. Export ────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.5)
plt.savefig("language_map.pdf", bbox_inches="tight", dpi=600, format="pdf")
plt.savefig("language_map.png", bbox_inches="tight", dpi=600, format="png")
print("Saved: language_map.pdf  (include in LaTeX with \\includegraphics)")
print("Saved: language_map.png  (600 dpi preview)")
plt.show()