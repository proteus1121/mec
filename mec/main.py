"""
ШВВП 2×1.5 мм² — Розрахунок параметрів передачі
Точне відтворення ШВВП_МБЕ-Т_е-вар.mcd (MathCAD)

═══════════════════════════════════════════════════
ФОРМУЛИ (точно за MCD):

Параметри кабелю:
  n  = 30   (кількість дротів у жилі)
  d_wire = 0.238 мм   (діаметр одного дроту)
  d1 = 1.53 мм        (фізичний діаметр жили = sqrt(n)*d_wire)
  Liz1 = 0.5 мм       (товщина ізоляції)
  lengthk1 = 5.06 мм  (ширина кабелю)
  a1 = 2.53 мм        (відстань між центрами = з формули геометрії)
  r1 = d1/2 = 0.765 мм
  psi = 0.6            (дає C1≈209 нФ/км точно як у MCD)
  ρ = 0.0175 Ом·мм²/м

krml(f) = 0.0105 · d1[мм] · √f     ← параметр від діаметра жили і частоти

Табличні функції (F, H, Q — від x; G — від xg):
  Ці функції описують поправки на скін-ефект та близькість проводів

Первинні параметри (в одиницях /км):
  R1(krml) = R01·[1 + F(krml) + ρ·G(krml)·(d1/a1)² / (1−H(krml)·(d1/a1)²)]
  L1(krml) = [4·ln(2·a1/d1) + μ·Q(krml)] · 10⁻⁴   [Гн/км]
  C1(f)    = ε(f)·10⁻⁶ / (36·ln(a1·ψ/r1))           [Ф/км]
  G1(f)    = ω·C1(f)·0.018                            [Сім/км]

де ε(f) = 2.45 + 2.73·2^(−1.54·10⁻⁷·f)  ← частотнозалежна діелектрична проникність

Вторинні параметри:
  α₀₁dB = 8.686·√(½·(|Z|·|Y| − (L1·C1·ω²−R1·G1)))   [дБ/км]
  β₀₁   = √(½·(|Z|·|Y| + (L1·C1·ω²−R1·G1)))           [рад/км]
  Z1     = √((R1+jωL1)/(G1+jωC1))                       [Ом]
  φ1     = arctg(Im(Z1)/Re(Z1))                         [рад]

Встановлення: pip install numpy matplotlib scipy
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.interpolate import interp1d
import warnings
from concurrent.futures import ThreadPoolExecutor
import threading
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════
#  Табличні дані функцій F, H, Q, G (з MCD image5)
# ═══════════════════════════════════════════════
VX  = np.array([0,    0.5,     1,       1.5,    2,      2.5,    3,     3.5,   4,     4.5,   5,     7,     10   ])
VF  = np.array([0,    3.26e-4, 3.19e-3, 0.0258, 0.0782, 0.1756, 0.318, 0.492, 0.678, 0.862, 1.042, 1.743, 2.799])
VH  = np.array([0.0417,0.042,  0.053,   0.092,  0.169,  0.263,  0.348, 0.416, 0.466, 0.503, 0.530, 0.596, 0.643])
VQ  = np.array([1,    0.9998,  0.997,   0.987,  0.961,  0.913,  0.845, 0.766, 0.686, 0.616, 0.556, 0.400, 0.286])
VXG = np.array([0.5,  1,       1.5,     2,      2.5,    3,      3.5,   4,     4.5,   5,     7,     10   ])
VG  = np.array([9.75e-4,0.01519,0.0691, 0.1724, 0.295,  0.405,  0.499, 0.584, 0.669, 0.755, 1.109, 1.641])

_iF = interp1d(VX,  VF, kind='linear', bounds_error=False, fill_value=(0, None))
_iH = interp1d(VX,  VH, kind='linear', bounds_error=False, fill_value=(0.0417, 0.643))
_iQ = interp1d(VX,  VQ, kind='linear', bounds_error=False, fill_value=(1.0,    None))
_iG = interp1d(VXG, VG, kind='linear', bounds_error=False, fill_value=(0,      None))

def F_fn(x):
    """F(x) — ефект скін-ефекту (активна складова).
    Для x<=10: табличні дані MCD.
    Для x>10: фізична асимптотика повного скін-ефекту F ~ x/(2√2) - 1
    (формула ((√2·x-3)^4)/4 з MCD — апроксимація перехідної зони, для x>>10 дає хибні значення ~10^10)
    """
    x = np.asarray(x, dtype=float)
    r = np.where(x <= 0.4, 0.0,
        np.where(x <= 10,  _iF(np.clip(x, 0, 10)),
                            x / (2.0 * np.sqrt(2)) - 1.0))
    return r

def H_fn(x):
    """H(x) — ефект близькості (активна)"""
    x = np.asarray(x, dtype=float)
    r = np.where(x < 0.5,  0.0417,
        np.where(x <= 10,  _iH(np.clip(x, 0, 10)),
                            0.75))
    return r

def Q_fn(x):
    """Q(x) — ефект близькості (реактивна)"""
    x = np.asarray(x, dtype=float)
    r = np.where(x < 0.7,  1.0,
        np.where(x <= 10,  _iQ(np.clip(x, 0, 10)),
                            2*np.sqrt(2)/np.where(x>0, x, 1e-10)))
    return r

def G_fn(xg):
    """G(xg) — ефект близькості (активна)"""
    xg = np.asarray(xg, dtype=float)
    r = np.where(xg < 0.5,  xg**2 / 64,
        np.where(xg <= 10,  _iG(np.clip(xg, 0.5, 10)),
                             np.sqrt(2)*(xg - 1)*0.125))
    return r


# ═══════════════════════════════════════════════
#  Розрахункові функції
# ═══════════════════════════════════════════════

def eps_f(f):
    """Частотнозалежна діелектрична проникність ПВХ (image3)"""
    return 2.45 + 2.73 * 2**(-1.54e-7 * f)

def krml_f(f, d1_mm=1.53):
    """krml(f) — параметр скін-ефекту (image5)"""
    return 0.0105 * d1_mm * np.sqrt(f)

def calc_primary(f_arr, params):
    """
    Розрахунок первинних параметрів R1, L1, C1, G1 (в /км)
    для масиву частот f_arr.
    """
    d1  = params['d1_mm']
    a1  = params['a1_mm']
    r1  = params['r1_mm']
    psi = params['psi']
    mu  = params['mu']
    rho = params['rho']
    R01 = params['R01']

    f = np.asarray(f_arr, dtype=float)
    f_safe = np.where(f < 1, 1, f)

    krml = krml_f(f_safe, d1)
    d_rat = (d1 / a1) ** 2

    F_v = F_fn(krml)
    H_v = H_fn(krml)
    Q_v = Q_fn(krml)
    G_v = G_fn(krml)

    denom_H = 1 - H_v * d_rat
    denom_H = np.where(np.abs(denom_H) < 1e-9, 1e-9, denom_H)

    R1 = R01 * (1 + F_v + rho/0.0175 * G_v * d_rat / denom_H)   # Ом/км
    L1 = (4 * np.log(2 * a1 / d1) + mu * Q_v) * 1e-4             # Гн/км
    C1 = eps_f(f_safe) * 1e-6 / (36 * np.log(a1 * psi / r1))    # Ф/км
    omega = 2 * np.pi * f_safe
    G1 = omega * C1 * 0.018                                        # Сім/км

    return R1, L1, C1, G1, krml

def calc_secondary(R1, L1, C1, G1, f_arr):
    """
    Вторинні параметри (image11, image14).
    Всі параметри в /км; результати:
      alpha01dB [дБ/км], beta01 [рад/км], Z1 [Ом], phi1 [рад]
    """
    f = np.asarray(f_arr, dtype=float)
    f_safe = np.where(f < 1, 1, f)
    omega = 2 * np.pi * f_safe

    # |Z|, |Y| — погонні імпеданс та адмітанс по модулю
    modZ = np.sqrt(R1**2 + (omega * L1)**2)
    modY = np.sqrt(G1**2 + (omega * C1)**2)

    # Перехресний член
    cross = L1 * C1 * omega**2 - R1 * G1

    # alpha (Нп/км) і beta (рад/км) — точна формула з image11
    inner = modZ * modY
    alpha01 = np.sqrt(np.maximum(0, 0.5 * (inner - cross)))   # Нп/км
    beta01  = np.sqrt(np.maximum(0, 0.5 * (inner + cross)))   # рад/км

    alpha01dB = 8.686 * alpha01   # дБ/км

    # Хвильовий опір Z1 = sqrt((R1+jwL1)/(G1+jwC1))
    Z_num = R1 + 1j * omega * L1
    Z_den = G1 + 1j * omega * C1
    Z_den_safe = np.where(np.abs(Z_den) < 1e-30, 1e-30 + 0j, Z_den)
    Z1 = np.sqrt(Z_num / Z_den_safe)

    # phi1 = arctg(Im(Z1)/Re(Z1))  [image14]
    phi1 = np.arctan2(np.imag(Z1), np.real(Z1))

    return alpha01dB, beta01, Z1, phi1


# ═══════════════════════════════════════════════
#  Кольори і шрифти
# ═══════════════════════════════════════════════
DARK  = "#1c1c2e"
MID   = "#16213e"
PANEL = "#0f1b30"
ACC   = "#e94560"
BLUE  = "#4fc3f7"
GREEN = "#69f0ae"
GOLD  = "#ffd54f"
PURP  = "#ce93d8"
ORNG  = "#ffb74d"
WHITE = "#e8eaf6"
GRAY  = "#78909c"

FS = 11   # базовий розмір шрифту графіків


# ═══════════════════════════════════════════════
#  Головний GUI
# ═══════════════════════════════════════════════

class CableApp:

    # Параметри за замовчуванням (з MCD image3, image5)
    DEFAULTS = {
        "n_wires":    30.0,    # кількість дротів у жилі
        "d_wire_mm":  0.238,   # діаметр одного дроту [мм]
        "Liz1_mm":    0.5,     # товщина ізоляції [мм]
        "lengthk1_mm":5.06,    # ширина кабелю [мм]
        "rho":        0.0175,  # питомий опір [Ом·мм²/м]
        "mu":         1.0,     # відносна магнітна проникність
        "psi":        0.6,    # коефіцієнт для ємності (дає C1≈209 нФ/км як у MCD)
    }
    STEP = 0.001
    _executor = ThreadPoolExecutor(max_workers=4)

    def __init__(self, root):
        self.root = root
        self.root.title("ШВВП 2×1.5 — Параметри передачі  |  МBЕ MCD")
        self.root.configure(bg=DARK)
        self.root.geometry("1680x990")
        self.root.minsize(1300, 750)
        self._last_data = None
        self._build_ui()
        self._start_calculate()

    # ─── Побудова інтерфейсу ────────────────────

    def _build_ui(self):
        # Ліва панель
        left = tk.Frame(self.root, bg=PANEL, width=318)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0), pady=6)
        left.pack_propagate(False)

        # Заголовок
        hdr = tk.Frame(left, bg=ACC, pady=8)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="⚡  ШВВП 2×1.5 мм²", bg=ACC, fg=WHITE,
                 font=("Segoe UI", 13, "bold")).pack()
        tk.Label(hdr, text="МБЕ  |  ШВВП_МБЕ-Т_е-вар.mcd", bg=ACC, fg="#ffe0e0",
                 font=("Segoe UI", 9)).pack()

        # Прокрутка
        cv = tk.Canvas(left, bg=PANEL, highlightthickness=0)
        sb = ttk.Scrollbar(left, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        cv.pack(fill=tk.BOTH, expand=True)
        inner = tk.Frame(cv, bg=PANEL)
        wid = cv.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfig(wid, width=e.width))
        cv.bind_all("<MouseWheel>",
                    lambda e: cv.yview_scroll(-1*(e.delta//120), "units"))

        sf = inner

        self.svars = {}

        # ── Блок: Конструкція кабелю ──
        self._sec(sf, "📐  КОНСТРУКЦІЯ КАБЕЛЮ  (ШВВП 2×1.5)")
        self._sl(sf, "Кількість дротів у жилі  n",
                 "n=30 дротів, кожна жила", "n_wires", 7, 61, step=1)
        self._sl(sf, "Діаметр одного дроту  d_wire  [мм]",
                 "diametrDrotiv = 0.238 мм (з MCD)", "d_wire_mm", 0.1, 0.8)
        self._sl(sf, "Товщина ізоляції  Liz1  [мм]",
                 "Liz1 = 0.5 мм (з MCD)", "Liz1_mm", 0.2, 2.0)
        self._sl(sf, "Ширина кабелю  lengthk1  [мм]",
                 "lengthk1 = 5.06 мм (з MCD)", "lengthk1_mm", 3.0, 12.0)

        # ── Блок: Матеріал ──
        self._sec(sf, "⚗️  МАТЕРІАЛ")

        # Приховані DoubleVar — оновлюються радіокнопками, читаються в calculate()
        self.svars["rho"] = tk.DoubleVar(value=self.DEFAULTS["rho"])
        self.svars["mu"]  = tk.DoubleVar(value=self.DEFAULTS["mu"])

        _MATS = {
            "copper":    dict(name="Мідь",     rho=0.0175, mu=1.0,
                              info="ρ = 0.0175 Ом·мм²/м\nμ = 1.0\nσ = 57.14 МСм/м"),
            "aluminium": dict(name="Алюміній", rho=0.0262, mu=1.0,
                              info="ρ = 0.0262 Ом·мм²/м\nμ = 1.0\nσ = 38.17 МСм/м"),
        }
        self._mat_key = tk.StringVar(value="copper")

        mat_frm = tk.Frame(sf, bg=MID, bd=1, relief=tk.RIDGE)
        mat_frm.pack(fill=tk.X, padx=6, pady=(0, 4))
        tk.Label(mat_frm, text="  Провідник  ", bg=BLUE, fg=DARK,
                 font=("Segoe UI", 9, "bold")).pack(fill=tk.X)
        rb_row = tk.Frame(mat_frm, bg=MID)
        rb_row.pack(fill=tk.X, padx=5, pady=4)
        self._mat_info = tk.Label(mat_frm, text=_MATS["copper"]["info"],
                                   bg="#0a1525", fg=GOLD,
                                   font=("Courier New", 9),
                                   justify=tk.LEFT, anchor="w", padx=8, pady=4)
        self._mat_info.pack(fill=tk.X, padx=5, pady=(0, 5))

        def _on_mat():
            m = _MATS[self._mat_key.get()]
            self.svars["rho"].set(m["rho"])
            self.svars["mu"].set(m["mu"])
            self._mat_info.config(text=m["info"])

        for _k, _m in _MATS.items():
            _c = GOLD if _k == "copper" else "#a0c4ff"
            tk.Radiobutton(rb_row, text=_m["name"],
                           variable=self._mat_key, value=_k,
                           command=_on_mat,
                           bg=MID, fg=_c, selectcolor="#0a1525",
                           font=("Segoe UI", 10, "bold"),
                           activebackground=MID, activeforeground=_c,
                           cursor="hand2").pack(side=tk.LEFT, padx=8)

        self._sl(sf, "Коефіцієнт ємності  ψ",
                 "ψ=0.6 → C1≈209 нФ/км (MCD)", "psi", 0.3, 1.5)

        # Кнопка
        self._calc_btn = tk.Button(sf, text="▶   РОЗРАХУВАТИ",
                  command=self._start_calculate,
                  bg=ACC, fg=WHITE, font=("Segoe UI", 11, "bold"),
                  relief=tk.FLAT, cursor="hand2", pady=8,
                  activebackground="#c0392b")
        self._calc_btn.pack(fill=tk.X, padx=6, pady=(10, 4))
        self._status_lbl = tk.Label(sf, text="", bg=PANEL, fg=GOLD,
                                     font=("Segoe UI", 8, "italic"))
        self._status_lbl.pack(fill=tk.X, padx=6)

        # Розрахована геометрія
        gf = tk.Frame(sf, bg=MID, bd=1, relief=tk.RIDGE)
        gf.pack(fill=tk.X, padx=6, pady=(0, 4))
        tk.Label(gf, text="  РОЗРАХОВАНА ГЕОМЕТРІЯ  ",
                 bg=BLUE, fg=DARK, font=("Segoe UI", 9, "bold")).pack(fill=tk.X)
        self.gl = {}
        for k, lbl in [("d1","d1 = "), ("a1","a1 = "),
                        ("r1","r1 = "), ("R01","R01= ")]:
            row = tk.Frame(gf, bg=MID)
            row.pack(fill=tk.X, padx=5, pady=1)
            tk.Label(row, text=lbl, bg=MID, fg=GRAY,
                     font=("Courier New", 10), width=5, anchor="w").pack(side=tk.LEFT)
            v = tk.Label(row, text="—", bg=MID, fg=BLUE,
                         font=("Courier New", 11))
            v.pack(side=tk.LEFT)
            self.gl[k] = v

        # Результати при вибраній частоті
        res = tk.Frame(sf, bg=MID, bd=1, relief=tk.RIDGE)
        res.pack(fill=tk.X, padx=6, pady=(0, 4))
        self._freq_title = tk.Label(res, text="  РЕЗУЛЬТАТИ при 24414 Гц  ",
                 bg=GREEN, fg=DARK, font=("Segoe UI", 9, "bold"))
        self._freq_title.pack(fill=tk.X)

        freq_frm = tk.Frame(res, bg=MID)
        freq_frm.pack(fill=tk.X, padx=5, pady=(4, 0))
        tk.Label(freq_frm, text="i-підканал:", bg=MID, fg=GRAY,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self._freq_var = tk.IntVar(value=1)
        self._freq_lbl = tk.Label(freq_frm, text="i=1  f=24 414 Гц",
                                   bg=MID, fg=GOLD,
                                   font=("Courier New", 8, "bold"))
        self._freq_lbl.pack(side=tk.RIGHT)

        def _on_freq(val):
            i = int(float(val))
            f_hz = max(1, i) * 24414 if i >= 1 else 1
            s = f"{f_hz:,}".replace(",", " ")
            self._freq_lbl.config(text=f"i={i}  f={s} Гц")
            self._freq_title.config(text=f"  РЕЗУЛЬТАТИ при {s} Гц  ")
            # Debounce: скасовуємо попередній виклик і відкладаємо на 40 мс
            if hasattr(self, "_slider_after_id"):
                self.root.after_cancel(self._slider_after_id)
            self._slider_after_id = self.root.after(
                40, lambda _i=i: self._update_results_for_freq(_i)
            )

        ttk.Scale(res, from_=0, to=4096, variable=self._freq_var,
                  orient=tk.HORIZONTAL, command=_on_freq
                  ).pack(fill=tk.X, padx=5, pady=(2, 5))

        self.rl = {}
        for k, lbl, unit in [
            ("krml",  "krml=", ""),
            ("R1",    "R1  =", "Ом/км"),
            ("L1",    "L1  =", "мкГн/км"),
            ("C1",    "C1  =", "нФ/км"),
            ("G1",    "G1  =", "мСм/км"),
            ("alpha", "α   =", "дБ/км"),
            ("beta",  "β   =", "рад/км"),
            ("Z1re",  "Re(Z1)=","Ом"),
            ("phi1",  "φ1  =", "рад"),
        ]:
            row = tk.Frame(res, bg=MID)
            row.pack(fill=tk.X, padx=5, pady=1)
            tk.Label(row, text=lbl, bg=MID, fg=GRAY,
                     font=("Courier New", 10), width=7, anchor="w").pack(side=tk.LEFT)
            v = tk.Label(row, text="—", bg=MID, fg=GREEN,
                         font=("Courier New", 10), width=10, anchor="w")
            v.pack(side=tk.LEFT)
            tk.Label(row, text=unit, bg=MID, fg=GRAY,
                     font=("Segoe UI", 9)).pack(side=tk.LEFT)
            self.rl[k] = v

        # Схема
        self._sec(sf, "🔵  ПЕРЕРІЗ КАБЕЛЮ")
        self.scheme = tk.Canvas(sf, width=295, height=110,
                                bg="#080e1c", highlightthickness=1,
                                highlightbackground=ACC)
        self.scheme.pack(padx=6, pady=(0, 6))

        # Права панель
        right = tk.Frame(self.root, bg=DARK)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.fig = Figure(facecolor=DARK)
        self.cfig = FigureCanvasTkAgg(self.fig, master=right)
        tb = NavigationToolbar2Tk(self.cfig, right, pack_toolbar=False)
        tb.update(); tb.configure(bg=MID)
        tb.pack(side=tk.BOTTOM, fill=tk.X)
        self.cfig.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _sec(self, p, title):
        f = tk.Frame(p, bg=MID, pady=3)
        f.pack(fill=tk.X, pady=(6, 1))
        tk.Label(f, text=f"  {title}", bg=MID, fg=GOLD,
                 font=("Segoe UI", 9, "bold"), anchor="w").pack(fill=tk.X)

    def _sl(self, parent, label, hint, key, mn, mx, step=None):
        default = self.DEFAULTS[key]
        st = step if step else self.STEP
        frm = tk.Frame(parent, bg=PANEL)
        frm.pack(fill=tk.X, padx=6, pady=2)
        tk.Label(frm, text=label, bg=PANEL, fg=WHITE,
                 font=("Segoe UI", 10), anchor="w").pack(fill=tk.X)
        if hint:
            tk.Label(frm, text="  " + hint, bg=PANEL, fg=GRAY,
                     font=("Segoe UI", 8, "italic"), anchor="w").pack(fill=tk.X)
        row = tk.Frame(frm, bg=PANEL)
        row.pack(fill=tk.X)
        var = tk.DoubleVar(value=default)
        self.svars[key] = var

        def snap(v, lo=mn, hi=mx, s=st):
            raw = var.get()
            snapped = round(round(raw / s) * s, 10)
            snapped = max(lo, min(hi, snapped))
            if abs(var.get() - snapped) > 1e-9:
                var.set(snapped)
            var._lbl.config(text=f"{snapped:.3f}")

        ttk.Scale(row, from_=mn, to=mx, variable=var,
                  orient=tk.HORIZONTAL, command=snap
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        lbl = tk.Label(row, text=f"{default:.3f}", bg=PANEL, fg=ACC,
                       font=("Courier New", 11, "bold"), width=7)
        lbl.pack(side=tk.LEFT, padx=3)
        var._lbl = lbl

    # ─── Розрахунок ────────────────────────────

    def _start_calculate(self):
        """Запускає розрахунок у фоновому потоці — UI не підвисає."""
        self._calc_btn.config(state=tk.DISABLED, text="⏳  Розрахунок...")
        self._status_lbl.config(text="")
        def _run():
            self.calculate()
            self.root.after(0, lambda: (
                self._calc_btn.config(state=tk.NORMAL, text="▶   РОЗРАХУВАТИ"),
                self._status_lbl.config(text="✓ Готово")
            ))
        threading.Thread(target=_run, daemon=True).start()

    def _update_results_for_freq(self, i_idx):
        """Оновлює ліву панель і маркери без перерахунку (з кешу)."""
        if self._last_data is None:
            return
        d  = self._last_data
        i1 = int(np.clip(np.argmin(np.abs(d["i_arr"] - i_idx)), 0, len(d["f_arr"])-1))
        self.rl["krml"].config( text=f"{d['krml'][i1]:.4f}")
        self.rl["R1"].config(   text=f"{d['R1'][i1]:.3f}")
        self.rl["L1"].config(   text=f"{d['L1'][i1]*1e4:.4f}·10⁻⁴")
        self.rl["C1"].config(   text=f"{d['C1'][i1]*1e9:.4f}")
        self.rl["G1"].config(   text=f"{d['G1'][i1]*1e3:.4f}")
        self.rl["alpha"].config(text=f"{d['alpha'][i1]:.4f}")
        self.rl["beta"].config( text=f"{d['beta'][i1]:.4f}")
        self.rl["Z1re"].config( text=f"{np.real(d['Z1'][i1]):.3f}")
        self.rl["phi1"].config( text=f"{d['phi1'][i1]:.4f}")
        self._refresh_markers(i1)

    def calculate(self):
        p = self.svars
        n     = max(1, int(round(p["n_wires"].get())))
        d_w   = p["d_wire_mm"].get()   # мм
        Liz1  = p["Liz1_mm"].get()     # мм
        lk1   = p["lengthk1_mm"].get() # мм
        rho   = p["rho"].get()
        mu    = p["mu"].get()
        psi   = p["psi"].get()

        # Геометрія (з MCD image5):
        # d1 = фізичний діаметр жили (мм) — оцінюється за n і d_wire
        d1 = np.sqrt(n) * d_w      # приблизний зовнішній діаметр жили [мм]
        # Точніше: для круглої жили з n дротів d1 ≈ d_wire * (1+1/sin(pi/n)) / 2... 
        # В MCD задано безпосередньо d1=1.53 мм
        # d1 = min(d1, 1.99 * d_w * np.sqrt(n / np.pi))

        # a1 = (lk1 - 2*d1 - Liz1*4) + d1 + 2*Liz1  # відстань між центрами [мм]
        # r1 = d1 / 2

        # # Захист від некоректних значень
        # if a1 <= d1:
        #     a1 = d1 * 1.2
        # if r1 <= 0:
        #     r1 = 0.1

        # TODO: Для точного відтворення MCD використовуємо задані там значення, а не формули
        d1 = 1.53
        a1 = 2.53
        r1 = d1 / 2

        # R01 (Ом/км) — DC опір двожильного проводу


        R01 = 2000 * 4 * rho / (d_w**2 * n * np.pi)

        # Показуємо геометрію
        self.gl["d1"].config(text=f"{d1:.4f} мм")
        self.gl["a1"].config(text=f"{a1:.4f} мм")
        self.gl["r1"].config(text=f"{r1:.4f} мм")
        self.gl["R01"].config(text=f"{R01:.4f} Ом/км")

        params = dict(d1_mm=d1, a1_mm=a1, r1_mm=r1, psi=psi,
                      mu=mu, rho=rho, R01=R01)

        # Частоти: OFDM підканали 24414 Гц × i, i=0..4096 (до ~100 МГц)
        # f_i = i*24414, але f_0=1 (щоб уникнути нуля)
        N_pts = 500
        i_arr = np.linspace(0, 4096, N_pts)
        f_arr = np.where(i_arr < 1, 1, i_arr * 24414)

        try:
            N = len(f_arr)
            segs = [f_arr[i*N//4:(i+1)*N//4] for i in range(4)]

            def _calc_seg(f_seg):
                r, l, c, g, k = calc_primary(f_seg, params)
                a, b, z, p    = calc_secondary(r, l, c, g, f_seg)
                return r, l, c, g, k, a, b, z, p

            futures = [self._executor.submit(_calc_seg, seg) for seg in segs]
            res     = [f.result() for f in futures]

            R1    = np.concatenate([r[0] for r in res])
            L1    = np.concatenate([r[1] for r in res])
            C1    = np.concatenate([r[2] for r in res])
            G1    = np.concatenate([r[3] for r in res])
            krml  = np.concatenate([r[4] for r in res])
            alpha = np.concatenate([r[5] for r in res])
            beta  = np.concatenate([r[6] for r in res])
            Z1    = np.concatenate([r[7] for r in res])
            phi1  = np.concatenate([r[8] for r in res])
        except Exception:
            import traceback; traceback.print_exc()
            return

        # Результати при вибраній частоті (слайдер i-підканалу)
        i_sel = self._freq_var.get() if hasattr(self, "_freq_var") else 1
        i1 = int(np.clip(np.argmin(np.abs(i_arr - i_sel)), 0, len(f_arr)-1))
        f_sel = f_arr[i1]
        if hasattr(self, "_freq_title"):
            s = f"{int(f_sel):,}".replace(",", " ")
            self._freq_title.config(text=f"  РЕЗУЛЬТАТИ при {s} Гц  ")
            self._freq_lbl.config(text=f"i={i_sel}  f={s} Гц")
        self.rl["krml"].config( text=f"{krml[i1]:.4f}")
        self.rl["R1"].config(   text=f"{R1[i1]:.3f}")
        self.rl["L1"].config(   text=f"{L1[i1]*1e4:.4f}·10⁻⁴")
        self.rl["C1"].config(   text=f"{C1[i1]*1e9:.4f}")
        self.rl["G1"].config(   text=f"{G1[i1]*1e3:.4f}")
        self.rl["alpha"].config(text=f"{alpha[i1]:.4f}")
        self.rl["beta"].config( text=f"{beta[i1]:.4f}")
        self.rl["Z1re"].config( text=f"{np.real(Z1[i1]):.3f}")
        self.rl["phi1"].config( text=f"{phi1[i1]:.4f}")

        self._last_data = dict(i_arr=i_arr, f_arr=f_arr,
                               R1=R1, L1=L1, C1=C1, G1=G1,
                               alpha=alpha, beta=beta, Z1=Z1, phi1=phi1,
                               krml=krml, params=params)
        self._draw(i_arr, f_arr, R1, L1, C1, G1, alpha, beta, Z1, phi1, krml, params, i1)
        self._draw_scheme(d1, r1, a1, int(round(p["n_wires"].get())))

    # ─── 9 графіків ─────────────────────────────

    def _draw(self, i_arr, f, R1, L1, C1, G1, alpha, beta, Z1, phi1, krml, params, i1):
        self.fig.clear()
        self.fig.patch.set_facecolor(DARK)
        # Словник для динамічних художників (маркери, анотації, vlines)
        dyn = {}

        # 4 рядки × 4 стовпці = 13 графіків
        # Рядок 0: ε(f)  | F(x)   | H(x)      | —
        # Рядок 1: G(xg) | Q(x)   | R1(f)     | —
        # Рядок 2: L1(f) | C1(f)  | G1(f)     | —
        # Рядок 3: α(f)  | β(f)   | Re(Z1)(f) | φ1(f)
        gs = self.fig.add_gridspec(
            4, 4,
            hspace=0.55, wspace=0.42,
            left=0.05, right=0.98,
            top=0.94, bottom=0.05)

        f_plot = f
        d1 = params['d1_mm']

        self.fig.suptitle(
            f"ШВВП 2×1.5  |  d1={d1:.3f}мм  a1={params['a1_mm']:.3f}мм"
            f"  ρ={params['rho']:.4f}  μ={params['mu']:.1f}  ψ={params['psi']:.3f}",
            color=ACC, fontsize=11, fontweight="bold", fontfamily="monospace")

        def make_ax(row, col, ylabel, title, ycolor=WHITE, xlabel="f_i, Гц"):
            ax = self.fig.add_subplot(gs[row, col])
            ax.set_facecolor(MID)
            ax.tick_params(colors=WHITE, labelsize=FS-1, which='both')
            ax.set_xlabel(xlabel, fontsize=FS, color=WHITE)
            ax.set_ylabel(ylabel, fontsize=FS, color=ycolor)
            ax.set_title(title, color=BLUE, fontsize=FS, fontweight="bold", pad=4)
            ax.grid(True, color="#1e2e46", lw=0.6)
            ax.minorticks_on()
            ax.grid(True, which='minor', color="#171f30", lw=0.3)
            for sp in ax.spines.values():
                sp.set_color("#2a3a5c")
            return ax

        def make_fax(row, col, ylabel, title, ycolor=WHITE):
            """Графік від частоти з xlim"""
            ax = make_ax(row, col, ylabel, title, ycolor)
            ax.set_xlim(0, f_plot[-1])
            return ax

        def pt(ax, xi, yi, fmt="%.4f", key=None):
            """Малює маркер + анотацію; якщо key — зберігає посилання в dyn."""
            dot, = ax.plot(xi, yi, 'o', color=GOLD, ms=5, zorder=6)
            ann = ax.annotate(fmt % yi, xy=(xi, yi), color=GOLD, fontsize=8,
                              xytext=(xi * 1.04, yi))
            if key:
                dyn[key + '_dot'] = dot
                dyn[key + '_ann'] = ann
                dyn[key + '_fmt'] = fmt
                dyn[key + '_ax']  = ax

        # ═══════════════════════════════════════════════════
        # РЯД 0
        # ═══════════════════════════════════════════════════

        # ГРАФІК 1 [0,0]: ε(f) = s_i
        ax = make_fax(0, 0, "s_i", "ε(f) - частотнозалежна діелектрична проникність", ACC)
        eps_arr = eps_f(f_plot)
        ax.plot(f_plot, eps_arr, color=ACC, lw=2.0)
        ax.annotate(f"{eps_arr[0]:.2f}",
                    xy=(f_plot[1], eps_arr[0]), color=GOLD, fontsize=9,
                    xytext=(f_plot[-1]*0.02, eps_arr[0]+0.04))
        ax.annotate(f"{eps_arr[-1]:.2f}",
                    xy=(f_plot[-1]*0.75, eps_arr[-1]), color=GOLD, fontsize=9,
                    xytext=(f_plot[-1]*0.55, eps_arr[-1]-0.15))

        # ГРАФІК 2 [0,1]: F(x)
        x_d = np.linspace(0, 10, 400)
        ax = make_ax(0, 1, "F(x)", "F(x)  —  linterp(vx, vf, x)", ACC, xlabel="x")
        ax.plot(x_d, F_fn(x_d), color=ACC, lw=2.0)
        ax.scatter(VX, VF, color=WHITE, s=18, zorder=5)
        vl_F = ax.axvline(krml[i1], color=GOLD, lw=0.9, ls="--", alpha=0.8,
                   label=f"krml={krml[i1]:.3f}")
        leg_F = ax.legend(fontsize=8, facecolor=MID, labelcolor=WHITE)
        dyn['vl_F'] = vl_F
        dyn['leg_F'] = leg_F
        dyn['ax_F']  = ax

        # ГРАФІК 3 [0,2]: H(x)
        ax = make_ax(0, 2, "H(x)", "H(x)  —  linterp(vx, vh, x)", BLUE, xlabel="x")
        ax.plot(x_d, H_fn(x_d), color=BLUE, lw=2.0)
        ax.scatter(VX, VH, color=WHITE, s=18, zorder=5)
        dyn['vl_H'] = ax.axvline(krml[i1], color=GOLD, lw=0.9, ls="--", alpha=0.8)

        # ═══════════════════════════════════════════════════
        # РЯД 1
        # ═══════════════════════════════════════════════════

        # ГРАФІК 4 [1,0]: G(xg)
        xg_d = np.linspace(0.1, 10, 400)
        ax = make_ax(1, 0, "G(xg)", "G(xg)  —  linterp(vxg, vg, xg)", GOLD, xlabel="xg")
        ax.plot(xg_d, G_fn(xg_d), color=GOLD, lw=2.0)
        ax.scatter(VXG, VG, color=WHITE, s=18, zorder=5)
        dyn['vl_G'] = ax.axvline(krml[i1], color=GOLD, lw=0.9, ls="--", alpha=0.4)

        # ГРАФІК 5 [1,1]: Q(x)
        ax = make_ax(1, 1, "Q(x)", "Q(x)  —  linterp(vx, vq, x)", GREEN, xlabel="x")
        ax.plot(x_d, Q_fn(x_d), color=GREEN, lw=2.0)
        ax.scatter(VX, VQ, color=WHITE, s=18, zorder=5)
        dyn['vl_Q'] = ax.axvline(krml[i1], color=GOLD, lw=0.9, ls="--", alpha=0.8)

        # ГРАФІК 6 [1,2]: R1(f)
        ax = make_fax(1, 2, "Ом/км", "R1(f)  —  Погонний опір", ACC)
        ax.plot(f_plot, R1, color=ACC, lw=2.0, label="R1(krml₁)")
        ax.axhline(params['R01'], color=GOLD, lw=1.0, ls=":",
                   label=f"R01={params['R01']:.2f} (DC)")
        pt(ax, f_plot[i1], R1[i1], key='R1')
        ax.legend(fontsize=8, facecolor=MID, labelcolor=WHITE)

        # ═══════════════════════════════════════════════════
        # РЯД 2
        # ═══════════════════════════════════════════════════

        # ГРАФІК 7 [2,0]: L1(f)
        ax = make_fax(2, 0, "L1 × 10⁻⁴  [Гн/км]", "L1(f)  —  Погонна індуктивність", BLUE)
        ax.plot(f_plot, L1 * 1e4, color=BLUE, lw=2.0, label="L1×10⁻⁴")
        pt(ax, f_plot[i1], L1[i1]*1e4, "%.4f", key='L1')
        ax.legend(fontsize=9, facecolor=MID, labelcolor=WHITE)

        # ГРАФІК 8 [2,1]: C1(f)
        ax = make_fax(2, 1, "C1  [нФ/км]", "C1(f)  —  Погонна ємність", GREEN)
        ax.plot(f_plot, C1 * 1e9, color=GREEN, lw=2.0, label="C1 [нФ/км]")
        pt(ax, f_plot[i1], C1[i1]*1e9, "%.4f", key='C1')
        ax.legend(fontsize=8, facecolor=MID, labelcolor=WHITE)

        # ГРАФІК 9 [2,2]: G1(f)
        ax = make_fax(2, 2, "G1  [мСм/км]", "G1(f)  —  Погонна провідність", ORNG)
        ax.plot(f_plot, G1 * 1e3, color=ORNG, lw=2.0, label="G1 = ω·C1·0.018")
        pt(ax, f_plot[i1], G1[i1]*1e3, "%.4f", key='G1')
        ax.legend(fontsize=9, facecolor=MID, labelcolor=WHITE)

        # ═══════════════════════════════════════════════════
        # РЯД 3
        # ═══════════════════════════════════════════════════

        # ГРАФІК 10 [3,0]: α01dB(f)
        ax = make_fax(3, 0, "α, дБ/км", "α01dB(f)  —  Загасання", ACC)
        ax.plot(f_plot, alpha, color=ACC, lw=2.0, label="α01dB")
        pt(ax, f_plot[i1], alpha[i1], "%.4f", key='alpha')
        ax.legend(fontsize=9, facecolor=MID, labelcolor=WHITE)

        # ГРАФІК 11 [3,1]: β01(f)
        ax = make_fax(3, 1, "β, рад/км", "β01(f)  —  Фазова характеристика", BLUE)
        ax.plot(f_plot, beta, color=BLUE, lw=2.0, label="β01")
        pt(ax, f_plot[i1], beta[i1], "%.4f", key='beta')
        ax.legend(fontsize=9, facecolor=MID, labelcolor=WHITE)

        # ═══════════════════════════════════
        # ГРАФІК 12 [3,2]: Re(Z1)(f)
        # X: 0 … 1×10^8 Гц
        # ═══════════════════════════════════

        ReZ = np.real(Z1)
        F_MAX = 1e8

        ax = make_fax(
            3, 2,
            "Re(Z1), Ом",
            "Re(Z1)(f)  —  Хвильовий опір",
            GREEN
        )

        ax.set_xlim(0, F_MAX)
        ax.set_ylim(40, 200)
        ax.plot(f_plot, ReZ, color=GREEN, lw=2.0)

        # Точка при вибраній частоті
        rez_dot, = ax.plot(f_plot[i1], ReZ[i1], 'o', color=GOLD, ms=6, zorder=6)
        rez_ann = ax.annotate(f"{ReZ[i1]:.3f} Ом",
                    xy=(f_plot[i1], ReZ[i1]),
                    xytext=(f_plot[i1] + F_MAX*0.03, ReZ[i1]),
                    color=GOLD, fontsize=8)
        dyn['ReZ_dot'] = rez_dot
        dyn['ReZ_ann'] = rez_ann
        dyn['ReZ_FMAX'] = F_MAX

        # Високочастотна асимптота
        Zhf = ReZ[-1]
        ax.axhline(Zhf, color=GOLD, lw=0.8, ls=':', alpha=0.7)
        ax.annotate(f"Z_hf={Zhf:.1f} Ом",
                    xy=(F_MAX*0.6, Zhf), xytext=(F_MAX*0.6, Zhf + 3),
                    color=GOLD, fontsize=8)

        # ═══════════════════════════════════════
        # ГРАФІК 13 [3,3]: φ1 — як у Mathcad
        # ═══════════════════════════════════════

        ax = make_ax(
            3, 3,
            "φ1, рад",
            "φ1(i)  —  arctg(Im(Z1)/Re(Z1))",
            GOLD,
            xlabel="i"
        )

        ax.set_xlim(0, 4096)
        ax.plot(i_arr, phi1, color=GOLD, lw=2.0)

        ax.axhline(0, color=GRAY, lw=0.8, ls='--')

        phi1_dot, = ax.plot(i_arr[i1], phi1[i1], 'o', color=GREEN, ms=6)
        phi1_ann = ax.annotate(f"{phi1[i1]:.4f}",
                    xy=(i_arr[i1], phi1[i1]),
                    xytext=(i_arr[i1] + 120, phi1[i1]),
                    color=GREEN,
                    fontsize=8)
        dyn['phi1_dot'] = phi1_dot
        dyn['phi1_ann'] = phi1_ann

        self._dyn_artists = dyn
        self.cfig.draw()

    def _refresh_markers(self, i1):
        """Швидке оновлення маркерів без повного перемальовування графіків."""
        if not hasattr(self, '_dyn_artists') or not self._dyn_artists:
            return
        if self._last_data is None:
            return
        d = self._last_data
        f = d['f_arr']; krml = d['krml']
        R1 = d['R1']; L1 = d['L1']; C1 = d['C1']; G1 = d['G1']
        alpha = d['alpha']; beta = d['beta']
        Z1 = d['Z1']; phi1 = d['phi1']
        i_arr = d['i_arr']
        ReZ = np.real(Z1)
        dyn = self._dyn_artists

        # Оновлюємо pt()-маркери: dot + annotation
        pt_data = {
            'R1':    (f[i1], R1[i1]),
            'L1':    (f[i1], L1[i1]*1e4),
            'C1':    (f[i1], C1[i1]*1e9),
            'G1':    (f[i1], G1[i1]*1e3),
            'alpha': (f[i1], alpha[i1]),
            'beta':  (f[i1], beta[i1]),
        }
        for key, (xi, yi) in pt_data.items():
            dot = dyn.get(key + '_dot')
            ann = dyn.get(key + '_ann')
            fmt = dyn.get(key + '_fmt', '%.4f')
            if dot is not None:
                dot.set_data([xi], [yi])
            if ann is not None:
                ann.xy = (xi, yi)
                ann.set_position((xi * 1.04, yi))
                ann.set_text(fmt % yi)

        # Re(Z1) маркер
        F_MAX = dyn.get('ReZ_FMAX', 1e8)
        rez_dot = dyn.get('ReZ_dot')
        rez_ann = dyn.get('ReZ_ann')
        if rez_dot is not None:
            rez_dot.set_data([f[i1]], [ReZ[i1]])
        if rez_ann is not None:
            rez_ann.xy = (f[i1], ReZ[i1])
            rez_ann.set_position((f[i1] + F_MAX*0.03, ReZ[i1]))
            rez_ann.set_text(f"{ReZ[i1]:.3f} Ом")

        # phi1 маркер
        phi1_dot = dyn.get('phi1_dot')
        phi1_ann = dyn.get('phi1_ann')
        if phi1_dot is not None:
            phi1_dot.set_data([i_arr[i1]], [phi1[i1]])
        if phi1_ann is not None:
            phi1_ann.xy = (i_arr[i1], phi1[i1])
            phi1_ann.set_position((i_arr[i1] + 120, phi1[i1]))
            phi1_ann.set_text(f"{phi1[i1]:.4f}")

        # Вертикальні лінії krml на F/H/G/Q графіках
        kv = krml[i1]
        for key in ('vl_F', 'vl_H', 'vl_G', 'vl_Q'):
            vl = dyn.get(key)
            if vl is not None:
                vl.set_xdata([kv, kv])

        # Оновлюємо підпис у легенді F(x)
        leg_F = dyn.get('leg_F')
        if leg_F is not None:
            texts = leg_F.get_texts()
            if texts:
                texts[0].set_text(f"krml={kv:.3f}")

        self.cfig.draw_idle()

    # ─── Схема перерізу ──────────────────────────

    def _draw_scheme(self, d1, r1, a1, n):
        c = self.scheme
        c.delete("all")
        W, H = 295, 110
        cx1, cx2, cy = W*0.28, W*0.72, H*0.50
        r_ins_mm = (a1/2 - r1) + r1  # просто r_ins ≈ a1/2
        sc = min((W*0.18) / max(a1/2, 0.1), (H*0.38) / max(a1/2, 0.1))
        rw_px = r1 * sc
        ri_px = (a1 / 2) * sc

        # ПВХ оболонка
        c.create_rectangle(cx1-ri_px-3, cy-ri_px-3, cx2+ri_px+3, cy+ri_px+3,
                           outline=ACC, width=2, fill="#08101e", dash=(5, 3))
        # МБЕ точки
        angles = np.linspace(0, 2*np.pi, min(n, 36), endpoint=False)
        for cx, fill_c, lbl in [(cx1, "#cc3333", "+"), (cx2, "#3355cc", "−")]:
            c.create_oval(cx-ri_px, cy-ri_px, cx+ri_px, cy+ri_px,
                          fill="#1a2a50", outline=BLUE, width=1)
            for a in angles:
                ex = cx + (rw_px + 3) * np.cos(a)
                ey = cy + (rw_px + 3) * np.sin(a)
                c.create_oval(ex-2, ey-2, ex+2, ey+2, fill=GRAY, outline="")
            c.create_oval(cx-rw_px, cy-rw_px, cx+rw_px, cy+rw_px,
                          fill=fill_c, outline=WHITE, width=1)
            c.create_text(cx, cy, text=lbl, fill=WHITE,
                          font=("Arial", max(7, int(rw_px*0.7)+5), "bold"))
        # Розмірна лінія
        yd = cy + ri_px + 14
        c.create_line(cx1, yd, cx2, yd, fill=WHITE, arrow=tk.BOTH)
        c.create_text((cx1+cx2)/2, yd+9,
                      text=f"a1={a1:.2f}мм  d1={d1:.2f}мм",
                      fill=WHITE, font=("Segoe UI", 9, "bold"))
        c.create_text(cx1, cy-ri_px-8,
                      text=f"r1={r1:.3f}мм",
                      fill=GRAY, font=("Segoe UI", 8))
        c.create_text(cx2, cy-ri_px-8,
                      text=f"n={n} дротів",
                      fill=GOLD, font=("Segoe UI", 8, "bold"))


# ═══════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    st = ttk.Style(root)
    st.theme_use("clam")
    st.configure("Horizontal.TScale",
                 background=PANEL, troughcolor="#253550",
                 sliderthickness=14, sliderrelief="flat")
    CableApp(root)
    root.mainloop()