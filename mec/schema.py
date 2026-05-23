"""
МБЕ / PLC  —  Мережевий Редактор та Аналізатор  (v2.0)
═══════════════════════════════════════════════════════
Візуальний редактор схеми + автоматичний аналіз топології
Метод ABCD-матриць (Частина 2: розгалужена мережа МБЕ)

• Малюйте вузли (TX, RX, Load, Node) та кабелі на схемі
• Задайте параметри кожного кабелю (L, α, β, Z₀) і навантаження (Zvid)
• Програма автоматично знаходить шлях TX→RX, виявляє відгалуження,
  перераховує Zvid→Zvidgal та будує каскад ABCD-матриць
• Результат: графік загасання [дБ] та фаза [рад] у частотному діапазоні
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from collections import deque

# ══════════════════════════════════════════════════════════════════════════════
#  КОЛЬОРИ
# ══════════════════════════════════════════════════════════════════════════════

BG      = '#0D1117'
PANEL   = '#161B22'
BORD    = '#30363D'
ACC     = '#58A6FF'
GRN     = '#3FB950'
RED     = '#F78166'
TEXT    = '#C9D1D9'
DIM     = '#8B949E'
ENTRY   = '#21262D'
BTN     = '#238636'
HDR     = '#1F2937'
CANV_BG = '#0A0E13'
GRID_C  = '#161B22'

NODE_CLR = {'plc_tx': '#2EA043', 'plc_rx': '#388BFD',
            'load':   '#DA3633', 'node':   '#3D444D'}
NODE_LBL = {'plc_tx': 'TX', 'plc_rx': 'RX', 'load': 'Z', 'node': '·'}
PLT      = {'bg': '#0D1117', 'axes': '#161B22', 'grid': '#30363D', 'text': '#C9D1D9'}
PALETTE  = ['#58A6FF', '#3FB950', '#F78166', '#D2A8FF', '#FFA657', '#79C0FF']

# ══════════════════════════════════════════════════════════════════════════════
#  МАТЕМАТИКА  (ABCD-матриці, лінії передачі)
# ══════════════════════════════════════════════════════════════════════════════

def abcd_line(gamma, Z0, L):
    """ABCD-матриця однорідної лінії: A=cosh(γL), B=Z0·sinh(γL), ..."""
    gl = gamma * L
    A = np.cosh(gl);  B = Z0 * np.sinh(gl)
    C = np.sinh(gl) / Z0
    return np.array([[A, B], [C, A]])


def Zin_branch(Z0b, gammab, Lb, Zvid_b):
    """
    Перерахунок Zvid → Zvidgal (вхідний опір відгалуження):
        Zvidgal = Z0·(Zvid·cosh(γL) + Z0·sinh(γL))
                     / (Z0·cosh(γL) + Zvid·sinh(γL))
    """
    gl  = gammab * Lb
    c, s = np.cosh(gl), np.sinh(gl)
    return Z0b * (Zvid_b * c + Z0b * s) / (Z0b * c + Zvid_b * s)


def abcd_shunt(Z):
    """ABCD-матриця паралельного шунта."""
    return np.array([[1. + 0j, 0j], [1. / Z, 1. + 0j]])


def transfer_H(M, Zs, ZL):
    """H = 2·ZL / (A·ZL + B + C·Zs·ZL + D·Zs)"""
    A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    return (2. * ZL) / (A * ZL + B + C * Zs * ZL + D * Zs)


# ══════════════════════════════════════════════════════════════════════════════
#  ГРАФ МЕРЕЖІ
# ══════════════════════════════════════════════════════════════════════════════

class NetworkGraph:
    """
    Граф PLC-мережі.
    Вузли (nodes): id → {type, x, y, name, Z}
      type  : 'plc_tx' | 'plc_rx' | 'load' | 'node'
      Z     : Zs для TX, ZL для RX, Zvid для load
    Ребра (edges): id → {n1, n2, L, alpha, beta, Z0}
    """

    def __init__(self):
        self.nodes: dict = {}
        self.edges: dict = {}
        self.adj:   dict = {}   # node_id → set of edge_ids
        self._nid = 0
        self._eid = 0

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def add_node(self, ntype, x, y, Z=75.0, name=None):
        nid = self._nid;  self._nid += 1
        pfx = {'plc_tx': 'TX', 'plc_rx': 'RX', 'load': 'Load', 'node': 'N'}
        self.nodes[nid] = {
            'type': ntype, 'x': x, 'y': y,
            'name': name or f"{pfx.get(ntype, 'N')}{nid}",
            'Z': Z
        }
        self.adj[nid] = set()
        return nid

    def add_edge(self, n1, n2, L=30., alpha=0.04, beta=0.18, Z0=150.):
        if n1 == n2:
            return None
        for eid, e in self.edges.items():
            if {e['n1'], e['n2']} == {n1, n2}:
                return None                          # вже з'єднані
        eid = self._eid;  self._eid += 1
        self.edges[eid] = {'n1': n1, 'n2': n2,
                           'L': L, 'alpha': alpha, 'beta': beta, 'Z0': Z0}
        self.adj[n1].add(eid)
        self.adj[n2].add(eid)
        return eid

    def remove_node(self, nid):
        if nid not in self.nodes:
            return
        for eid in list(self.adj.get(nid, set())):
            self.remove_edge(eid)
        del self.nodes[nid]
        self.adj.pop(nid, None)

    def remove_edge(self, eid):
        if eid not in self.edges:
            return
        e = self.edges[eid]
        self.adj[e['n1']].discard(eid)
        self.adj[e['n2']].discard(eid)
        del self.edges[eid]

    def other_end(self, eid, nid):
        e = self.edges[eid]
        return e['n2'] if e['n1'] == nid else e['n1']

    # ── ТОПОЛОГІЯ ─────────────────────────────────────────────────────────────

    def find_path(self, start, end):
        """BFS → (node_ids_list, edge_ids_list) або (None, None)."""
        if start == end:
            return [start], []
        q = deque([(start, [start], [])])
        vis = {start}
        while q:
            cur, ns, es = q.popleft()
            for eid in self.adj.get(cur, set()):
                nxt = self.other_end(eid, cur)
                if nxt in vis:
                    continue
                nn, ne = ns + [nxt], es + [eid]
                if nxt == end:
                    return nn, ne
                vis.add(nxt)
                q.append((nxt, nn, ne))
        return None, None

    def compute_Zin(self, node_id, coming_edge_id):
        """
        Рекурсивно обчислює вхідний опір на вузлі node_id
        (прийшовши з боку coming_edge_id).
        Це Zvidgal для відгалуджень.
        """
        node = self.nodes[node_id]
        if node['type'] in ('load', 'plc_rx'):
            return complex(node['Z'])
        other = [e for e in self.adj[node_id] if e != coming_edge_id]
        if not other:
            return 1e9 + 0j                          # розімкнений кінець
        Y = 0j
        for eid in other:
            e   = self.edges[eid]
            nxt = self.other_end(eid, node_id)
            Zn  = self.compute_Zin(nxt, eid)
            g   = complex(e['alpha'], e['beta'])
            Zi  = Zin_branch(e['Z0'], g, e['L'], Zn)
            Y  += 1.0 / Zi
        return (1.0 / Y) if abs(Y) > 1e-30 else 1e9 + 0j

    # ── ВАЛІДАЦІЯ ─────────────────────────────────────────────────────────────

    def validate(self):
        tx = [n for n, d in self.nodes.items() if d['type'] == 'plc_tx']
        rx = [n for n, d in self.nodes.items() if d['type'] == 'plc_rx']
        if len(tx) != 1:
            return False, f"Потрібен рівно 1 вузол PLC TX  (знайдено: {len(tx)})"
        if len(rx) != 1:
            return False, f"Потрібен рівно 1 вузол PLC RX  (знайдено: {len(rx)})"
        pn, _ = self.find_path(tx[0], rx[0])
        if pn is None:
            return False, "Шлях від TX до RX не знайдено  (мережа не з'єднана)"
        return True, "OK"

    # ── РОЗРАХУНОК КАСКАДУ ────────────────────────────────────────────────────

    def build_cascade(self, freq_arr):
        """
        Будує ABCD-каскад по мережі і повертає
        (att[дБ], phase[рад], path_nodes, path_edges).

        Алгоритм:
          1. BFS: знаходить основний шлях TX→RX
          2. На кожному вузлі шляху: паралельно підключає Zvidgal
             усіх відгалужень (рекурсивно через compute_Zin)
          3. M_total = Π(abcd_line × abcd_shunt)
          4. H = transfer_H(M_total, Zs, ZL)
        """
        tx = next(n for n, d in self.nodes.items() if d['type'] == 'plc_tx')
        rx = next(n for n, d in self.nodes.items() if d['type'] == 'plc_rx')
        Zs = complex(self.nodes[tx]['Z'])
        ZL = complex(self.nodes[rx]['Z'])

        path_nodes, path_edges = self.find_path(tx, rx)
        peset = set(path_edges)

        # Попередньо обчислюємо паралельні шунти в кожному вузлі
        # γ залежить від частоти → Zvidgal перераховується для кожної частоти
        # shunts will be computed per-frequency inside loop
        shunts_template = (path_edges, path_nodes)  # placeholder

        N   = len(freq_arr)
        att = np.zeros(N)
        phi = np.zeros(N)

        for fi in range(N):
            f = freq_arr[fi]

            # compute shunts for this frequency
            shunts = {}
            for k, eid in enumerate(path_edges):
                cur = path_nodes[k + 1]
                if k + 1 == len(path_nodes) - 1:
                    continue
                brs = [e for e in self.adj[cur] if e not in peset]
                if not brs:
                    continue
                Y = 0j
                for beid in brs:
                    be  = self.edges[beid]
                    nxt = self.other_end(beid, cur)
                    Zn  = self.compute_Zin(nxt, beid)
                    bg  = complex(be['alpha'], be['beta'] * f)
                    Zi  = Zin_branch(be['Z0'], bg, be['L'], Zn)
                    Y  += 1.0 / Zi
                shunts[k + 1] = (1.0 / Y) if abs(Y) > 1e-30 else 1e9 + 0j

            M = np.eye(2, dtype=complex)
            for k, eid in enumerate(path_edges):
                e = self.edges[eid]
                f = freq_arr[fi]
                g = complex(e['alpha'], e['beta'] * f)
                M = M @ abcd_line(g, e['Z0'], e['L'])
                if k + 1 in shunts:
                    M = M @ abcd_shunt(shunts[k + 1])
            H        = transfer_H(M, Zs, ZL)
            absH     = max(abs(H), 1e-300)
            att[fi]  = -20. * np.log10(absH)
            phi[fi]  = np.angle(H)

        return att, phi, path_nodes, path_edges

    # ── СЕРІАЛІЗАЦІЯ ──────────────────────────────────────────────────────────

    def to_dict(self):
        return {'nodes': self.nodes, 'edges': self.edges,
                '_nid': self._nid, '_eid': self._eid}

    def from_dict(self, d):
        self.nodes = {int(k): v for k, v in d['nodes'].items()}
        self.edges = {int(k): v for k, v in d['edges'].items()}
        self._nid  = d['_nid'];  self._eid = d['_eid']
        self.adj   = {nid: set() for nid in self.nodes}
        for eid, e in self.edges.items():
            self.adj[e['n1']].add(eid)
            self.adj[e['n2']].add(eid)


# ══════════════════════════════════════════════════════════════════════════════
#  КАНВА (графічний редактор схеми)
# ══════════════════════════════════════════════════════════════════════════════

class NetCanvas(tk.Canvas):
    NR        = 18          # радіус вузла
    SNAP      = 20          # крок прив'язки до сітки (px)
    SEL_COL   = '#FFD700'
    PATH_COL  = '#58A6FF'
    WIRE_COL  = '#3D444D'
    WIRE_W    = 2
    PATH_W    = 3

    def __init__(self, parent, graph, on_select_cb):
        super().__init__(parent, bg=CANV_BG, highlightthickness=0, bd=0)
        self.graph       = graph
        self.on_select   = on_select_cb
        self.mode        = 'select'
        self.node_type   = 'node'
        self.selected    = None         # ('node'|'edge', id)
        self._wire_src   = None         # id першого вузла при з'єднанні
        self._drag_nid   = None
        self._drag_ox    = 0
        self._drag_oy    = 0
        self._path_n     = set()        # вузли основного шляху (підсвічення)
        self._path_e     = set()
        self._temp_line  = None         # пунктир при малюванні кабелю
        self._mouse_xy   = (0, 0)

        self.bind('<Button-1>',          self._click)
        self.bind('<B1-Motion>',         self._drag)
        self.bind('<ButtonRelease-1>',   self._release)
        self.bind('<Motion>',            self._motion)
        self.bind('<Configure>',         lambda _: self.redraw())
        self.bind('<Delete>',            self._del_selected)
        self.bind('<BackSpace>',         self._del_selected)
        self.focus_set()

    # ── Публічний інтерфейс ────────────────────────────────────────────────────

    def set_mode(self, mode, node_type=None):
        self.mode      = mode
        self._wire_src = None
        self._drag_nid = None
        if node_type:
            self.node_type = node_type
        cursors = {'select': 'arrow', 'add_node': 'crosshair',
                   'add_wire': 'crosshair', 'delete': 'X_cursor'}
        self.configure(cursor=cursors.get(mode, 'arrow'))
        self.redraw()

    def highlight_path(self, pn, pe):
        self._path_n = set(pn) if pn else set()
        self._path_e = set(pe) if pe else set()
        self.redraw()

    def clear_path(self):
        self._path_n.clear();  self._path_e.clear()
        self.redraw()

    # ── Перемальовування ──────────────────────────────────────────────────────

    def redraw(self, *_):
        self.delete('all')
        self._draw_grid()
        for eid in list(self.graph.edges):
            self._draw_edge(eid)
        for nid in list(self.graph.nodes):
            self._draw_node(nid)
        # Підказка при додаванні кабелю
        if self._wire_src is not None and self.mode == 'add_wire':
            n  = self.graph.nodes[self._wire_src]
            r  = self.NR + 4
            self.create_oval(n['x']-r, n['y']-r, n['x']+r, n['y']+r,
                outline=self.SEL_COL, width=2, dash=(4, 3), tags='hint')
            # Тимчасова лінія до курсора
            mx, my = self._mouse_xy
            self.create_line(n['x'], n['y'], mx, my,
                fill=self.SEL_COL, width=1, dash=(6, 4), tags='hint')

    def _draw_grid(self):
        w = max(self.winfo_width(), 800)
        h = max(self.winfo_height(), 600)
        s = 40
        for x in range(0, w + s, s):
            self.create_line(x, 0, x, h, fill=GRID_C, width=1)
        for y in range(0, h + s, s):
            self.create_line(0, y, w, y, fill=GRID_C, width=1)

    def _draw_edge(self, eid):
        e  = self.graph.edges[eid]
        n1 = self.graph.nodes[e['n1']]
        n2 = self.graph.nodes[e['n2']]
        x1, y1, x2, y2 = n1['x'], n1['y'], n2['x'], n2['y']
        sel  = self.selected == ('edge', eid)
        path = eid in self._path_e
        col  = self.SEL_COL if sel else (self.PATH_COL if path else self.WIRE_COL)
        w    = self.WIRE_W + (2 if sel or path else 0)
        tag  = f'edge_{eid}'

        self.create_line(x1, y1, x2, y2, fill=col, width=w,
                         arrow='last', arrowshape=(10, 13, 4), tags=tag)

        # Мітка посередині
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        lbl = f"L={e['L']} м   Z₀={e['Z0']} Ω"
        # Фон для читабельності
        self.create_rectangle(mx - 52, my - 18, mx + 52, my - 6,
            fill=CANV_BG, outline='', tags=tag)
        self.create_text(mx, my - 12, text=lbl,
            fill=TEXT if sel else DIM,
            font=('Consolas', 7), tags=tag)
        # α/β
        lbl2 = f"α={e['alpha']}  β={e['beta']}"
        self.create_rectangle(mx - 46, my - 6, mx + 46, my + 6,
            fill=CANV_BG, outline='', tags=tag)
        self.create_text(mx, my, text=lbl2,
            fill=DIM, font=('Consolas', 6), tags=tag)

    def _draw_node(self, nid):
        n    = self.graph.nodes[nid]
        x, y = n['x'], n['y']
        r    = self.NR
        nt   = n['type']
        col  = NODE_CLR[nt]
        sel  = self.selected == ('node', nid)
        path = nid in self._path_n
        wsrc = self._wire_src == nid
        tag  = f'node_{nid}'

        out_col = self.SEL_COL if (sel or wsrc) else \
                  (self.PATH_COL if path else BORD)
        out_w   = 3 if (sel or wsrc or path) else 1

        if nt in ('plc_tx', 'plc_rx'):
            self.create_rectangle(x-r, y-r, x+r, y+r,
                fill=col, outline=out_col, width=out_w, tags=tag)
        elif nt == 'load':
            pts = [x, y-r, x+r, y+r, x-r, y+r]
            self.create_polygon(pts, fill=col, outline=out_col,
                width=out_w, smooth=False, tags=tag)
        else:
            self.create_oval(x-r, y-r, x+r, y+r,
                fill=col, outline=out_col, width=out_w, tags=tag)

        self.create_text(x, y, text=NODE_LBL[nt],
            fill='white', font=('Consolas', 9, 'bold'), tags=tag)
        self.create_text(x, y + r + 9, text=n['name'],
            fill=TEXT, font=('Consolas', 8), tags=tag)

        # Параметр Z під назвою
        z_lbl = {'plc_tx': f"Zs={n['Z']}Ω", 'plc_rx': f"ZL={n['Z']}Ω",
                 'load':   f"Zvid={n['Z']}Ω", 'node': ''}
        if z_lbl[nt]:
            self.create_text(x, y + r + 20, text=z_lbl[nt],
                fill=DIM, font=('Consolas', 7), tags=tag)

    # ── Взаємодія з мишею ─────────────────────────────────────────────────────

    @staticmethod
    def _snap(v, s=20):
        return round(v / s) * s

    def _click(self, ev):
        self.focus_set()
        x, y = ev.x, ev.y

        if self.mode == 'add_node':
            sx, sy = self._snap(x), self._snap(y)
            nid = self.graph.add_node(self.node_type, sx, sy)
            self.selected = ('node', nid)
            self.on_select('node', nid)
            self.redraw()

        elif self.mode == 'add_wire':
            hit = self._hit_node(x, y)
            if hit is None:
                self._wire_src = None
                self.redraw()
                return
            if self._wire_src is None:
                self._wire_src = hit
                self.redraw()
            else:
                if hit != self._wire_src:
                    eid = self.graph.add_edge(self._wire_src, hit)
                    if eid is not None:
                        self.selected = ('edge', eid)
                        self.on_select('edge', eid)
                self._wire_src = None
                self.redraw()

        elif self.mode == 'delete':
            hit_n = self._hit_node(x, y)
            if hit_n is not None:
                if self.selected and self.selected[1] == hit_n:
                    self.selected = None;  self.on_select(None, None)
                self.graph.remove_node(hit_n)
            else:
                hit_e = self._hit_edge(x, y)
                if hit_e is not None:
                    if self.selected and self.selected[1] == hit_e:
                        self.selected = None;  self.on_select(None, None)
                    self.graph.remove_edge(hit_e)
            self.clear_path()
            self.redraw()

        elif self.mode == 'select':
            hit_n = self._hit_node(x, y)
            if hit_n is not None:
                self.selected  = ('node', hit_n)
                self._drag_nid = hit_n
                self._drag_ox  = x - self.graph.nodes[hit_n]['x']
                self._drag_oy  = y - self.graph.nodes[hit_n]['y']
                self.on_select('node', hit_n)
            else:
                hit_e = self._hit_edge(x, y)
                if hit_e is not None:
                    self.selected = ('edge', hit_e)
                    self.on_select('edge', hit_e)
                else:
                    self.selected = None
                    self.on_select(None, None)
            self.redraw()

    def _drag(self, ev):
        self._mouse_xy = (ev.x, ev.y)
        if self.mode == 'select' and self._drag_nid is not None:
            nx = self._snap(ev.x - self._drag_ox)
            ny = self._snap(ev.y - self._drag_oy)
            self.graph.nodes[self._drag_nid]['x'] = nx
            self.graph.nodes[self._drag_nid]['y'] = ny
            self.redraw()
        elif self.mode == 'add_wire' and self._wire_src is not None:
            self.redraw()

    def _release(self, _):
        self._drag_nid = None

    def _motion(self, ev):
        self._mouse_xy = (ev.x, ev.y)
        if self.mode == 'add_wire' and self._wire_src is not None:
            self.redraw()

    def _del_selected(self, _):
        if self.selected is None:
            return
        kind, eid = self.selected
        self.selected = None
        self.on_select(None, None)
        if kind == 'node':
            self.graph.remove_node(eid)
        else:
            self.graph.remove_edge(eid)
        self.clear_path()
        self.redraw()

    # ── Хіт-тест ──────────────────────────────────────────────────────────────

    def _hit_node(self, x, y):
        for nid, n in self.graph.nodes.items():
            if abs(n['x'] - x) <= self.NR + 3 and abs(n['y'] - y) <= self.NR + 3:
                return nid
        return None

    def _hit_edge(self, x, y, tol=8):
        for eid, e in self.graph.edges.items():
            n1 = self.graph.nodes[e['n1']]
            n2 = self.graph.nodes[e['n2']]
            d  = _point_line_dist(x, y, n1['x'], n1['y'], n2['x'], n2['y'])
            if d < tol:
                return eid
        return None


def _point_line_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == dy == 0:
        return ((px-x1)**2 + (py-y1)**2) ** .5
    t = max(0., min(1., ((px-x1)*dx + (py-y1)*dy) / (dx*dx + dy*dy)))
    return ((px - x1 - t*dx)**2 + (py - y1 - t*dy)**2) ** .5


# ══════════════════════════════════════════════════════════════════════════════
#  ПАНЕЛЬ ВЛАСТИВОСТЕЙ
# ══════════════════════════════════════════════════════════════════════════════

class PropsPanel(tk.Frame):

    def __init__(self, parent, graph, on_change_cb):
        super().__init__(parent, bg=PANEL)
        self.graph     = graph
        self.on_change = on_change_cb
        self._vars     = {}
        self._content  = tk.Frame(self, bg=PANEL)
        self._content.pack(fill='both', expand=True, padx=10, pady=6)
        self._show_empty()

    def show(self, etype, eid):
        for w in self._content.winfo_children():
            w.destroy()
        self._vars.clear()
        if etype is None:
            self._show_empty()
        elif etype == 'node':
            self._show_node(eid)
        elif etype == 'edge':
            self._show_edge(eid)

    # ── Порожній стан ─────────────────────────────────────────────────────────

    def _show_empty(self):
        tk.Label(self._content,
            text="Виберіть вузол або кабель\n"
                 "для редагування параметрів",
            bg=PANEL, fg=DIM,
            font=('Consolas', 9, 'italic'),
            justify='center').pack(expand=True)

    # ── Вузол ─────────────────────────────────────────────────────────────────

    def _show_node(self, nid):
        n = self.graph.nodes[nid]
        tn = {'plc_tx': 'PLC Передавач  (TX)', 'plc_rx': 'PLC Приймач  (RX)',
              'load': 'Навантаження  (Load)', 'node': 'Вузол / Розгалуження'}
        self._hdr(f"ВУЗОЛ  —  {tn.get(n['type'], n['type'])}", NODE_CLR[n['type']])
        self._field("ID:", str(nid))

        v_name = tk.StringVar(value=n['name'])
        self._entry("Ім'я:", v_name)
        v_name.trace_add('write', lambda *_: self._set(n, 'name', v_name.get()))
        self._vars['name'] = v_name

        zl = {'plc_tx': "Zs — опір джерела [Ом]:",
              'plc_rx': "ZL — опір навант. [Ом]:",
              'load':   "Zvid — опір навант. [Ом]:",
              'node':   None}
        if zl[n['type']]:
            vz = tk.StringVar(value=str(n['Z']))
            self._entry(zl[n['type']], vz)
            vz.trace_add('write', lambda *_: self._set_float(n, 'Z', vz))
            self._vars['Z'] = vz

        tips = {
            'plc_tx': "Джерело сигналу МБЕ.\nZs = вхідний опір передавача.",
            'plc_rx': "Приймач сигналу МБЕ.\nZL = вхідний опір приймача.",
            'load':   "Навантаження на відгалуженні.\n"
                      "Zvid перераховується у Zvidgal\n"
                      "з урахуванням параметрів кабелю.",
            'node':   "Точка розгалуження мережі.\n"
                      "З'єднуйте кілька кабелів\n"
                      "через цей вузол.",
        }
        self._tip(tips.get(n['type'], ''))

    # ── Кабель ────────────────────────────────────────────────────────────────

    def _show_edge(self, eid):
        e  = self.graph.edges[eid]
        n1 = self.graph.nodes[e['n1']]['name']
        n2 = self.graph.nodes[e['n2']]['name']
        self._hdr("КАБЕЛЬ", ACC)
        self._field("З'єднання:", f"{n1}  ──▶  {n2}")

        for key, label, fmt in [
            ('L',     "L  — довжина [м]:",           e['L']),
            ('alpha', "α  — загасання [Нп/м]:",      e['alpha']),
            ('beta',  "β  — коеф. фази [рад/м]:",    e['beta']),
            ('Z0',    "Z₀ — хвильовий опір [Ом]:",   e['Z0']),
        ]:
            v = tk.StringVar(value=str(fmt))
            self._entry(label, v)
            v.trace_add('write', lambda *_, k=key, vv=v: self._set_float(e, k, vv))
            self._vars[key] = v

        self._tip("γ = α + j·β   [Нп/м + j·рад/м]\n"
                  "ABCD:  cosh(γL), Z₀·sinh(γL), ...\n"
                  "Матриця автоматично включається\nу каскад при розрахунку.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _hdr(self, text, color):
        tk.Label(self._content, text=text, bg=PANEL, fg=color,
            font=('Consolas', 9, 'bold')).pack(anchor='w', pady=(2, 6))
        tk.Frame(self._content, bg=BORD, height=1).pack(fill='x', pady=(0, 6))

    def _field(self, label, value):
        row = tk.Frame(self._content, bg=PANEL);  row.pack(fill='x', pady=1)
        tk.Label(row, text=label, bg=PANEL, fg=DIM,
            font=('Consolas', 8), width=24, anchor='w').pack(side='left')
        tk.Label(row, text=value, bg=PANEL, fg=TEXT,
            font=('Consolas', 8)).pack(side='left')

    def _entry(self, label, var):
        row = tk.Frame(self._content, bg=PANEL);  row.pack(fill='x', pady=2)
        tk.Label(row, text=label, bg=PANEL, fg=DIM,
            font=('Consolas', 8), width=24, anchor='w',
            wraplength=170, justify='left').pack(side='left')
        tk.Entry(row, textvariable=var, width=10,
            bg=ENTRY, fg=TEXT, insertbackground=TEXT,
            relief='flat', font=('Consolas', 9)).pack(side='left', padx=(4, 0))

    def _tip(self, text):
        if not text:
            return
        tk.Label(self._content, text=text, bg=PANEL, fg=DIM,
            font=('Consolas', 7, 'italic'), justify='left',
            wraplength=220).pack(anchor='w', pady=(8, 0))

    def _set(self, obj, key, val):
        obj[key] = val
        self.on_change()

    def _set_float(self, obj, key, var):
        try:
            obj[key] = float(var.get())
            self.on_change()
        except ValueError:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  ЛЕГЕНДА
# ══════════════════════════════════════════════════════════════════════════════

class Legend(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=PANEL)
        items = [
            (NODE_CLR['plc_tx'], '■', 'PLC TX  — передавач (Zs)'),
            (NODE_CLR['plc_rx'], '■', 'PLC RX  — приймач (ZL)'),
            (NODE_CLR['load'],   '▲', 'Load    — навантаження (Zvid)'),
            (NODE_CLR['node'],   '●', 'Node    — вузол/розгалуження'),
            (ACC,                '—', 'Основний шлях TX→RX'),
            (DIM,                '—', 'Кабель відгалуження'),
        ]
        for col, sym, lbl in items:
            row = tk.Frame(self, bg=PANEL);  row.pack(anchor='w', padx=8, pady=1)
            tk.Label(row, text=sym, bg=PANEL, fg=col,
                font=('Consolas', 10)).pack(side='left')
            tk.Label(row, text=lbl, bg=PANEL, fg=DIM,
                font=('Consolas', 8)).pack(side='left', padx=4)


# ══════════════════════════════════════════════════════════════════════════════
#  ГОЛОВНЕ ВІКНО
# ══════════════════════════════════════════════════════════════════════════════

class MBENetApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("МБЕ / PLC  —  Мережевий Редактор та Аналізатор")
        self.geometry("1560x950")
        self.minsize(1200, 720)
        self.configure(bg=BG)

        self.graph   = NetworkGraph()
        self._plot_n = 0
        self._fmin   = tk.DoubleVar(value=1.0)
        self._fmax   = tk.DoubleVar(value=30.0)
        self._fpts   = tk.IntVar(value=500)

        self._build_styles()
        self._build_ui()
        self._load_example()

    # ── ttk стилі ─────────────────────────────────────────────────────────────

    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use('clam')
        s.configure('.', background=BG, foreground=TEXT,
                    fieldbackground=ENTRY, bordercolor=BORD,
                    insertcolor=TEXT, relief='flat')
        s.configure('TFrame',     background=BG)
        s.configure('TLabel',     background=BG, foreground=TEXT)
        s.configure('TEntry',     fieldbackground=ENTRY, foreground=TEXT,
                    insertcolor=TEXT, relief='flat', borderwidth=1)
        s.configure('TLabelframe', background=BG, bordercolor=BORD, relief='groove')
        s.configure('TLabelframe.Label', background=BG, foreground=ACC,
                    font=('Consolas', 9, 'bold'))

    # ── UI layout ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Заголовок
        hdr = tk.Frame(self, bg=HDR, height=48)
        hdr.pack(fill='x');  hdr.pack_propagate(False)
        tk.Label(hdr,
            text="⚡  МБЕ / PLC  —  Мережевий Редактор та Аналізатор  "
                 "│  ABCD-матриці  │  Розгалужена мережа",
            bg=HDR, fg=ACC,
            font=('Consolas', 12, 'bold')).pack(side='left', padx=18, pady=12)

        # Основне тіло — СПОЧАТКУ, щоб canvas існував до toolbar
        body = tk.Frame(self, bg=BG)
        body.pack(fill='both', expand=True, padx=4, pady=(0, 4))

        # ── КАНВА — створюємо ПЕРШОЮ ──────────────────────────────────────────
        canv_frame = tk.Frame(body, bg=BORD, bd=1)
        canv_frame.pack(side='left', fill='both', expand=True)
        self.net_canvas = NetCanvas(canv_frame, self.graph, self._on_select)
        self.net_canvas.pack(fill='both', expand=True)

        # ── ПРАВА КОЛОНКА ─────────────────────────────────────────────────────
        right = tk.Frame(body, bg=BG, width=430)
        right.pack(side='left', fill='y', padx=(4, 0))
        right.pack_propagate(False)

        # Властивості
        pp_frame = tk.Frame(right, bg=PANEL, height=260)
        pp_frame.pack(fill='x');  pp_frame.pack_propagate(False)
        tk.Label(pp_frame, text="  ВЛАСТИВОСТІ", bg=PANEL, fg=ACC,
            font=('Consolas', 9, 'bold')).pack(anchor='w', padx=8, pady=(8, 2))
        tk.Frame(pp_frame, bg=BORD, height=1).pack(fill='x', padx=8)
        self.props = PropsPanel(pp_frame, self.graph, self.net_canvas.redraw)
        self.props.pack(fill='both', expand=True)

        # Легенда
        leg_frame = tk.Frame(right, bg=PANEL)
        leg_frame.pack(fill='x', pady=(4, 0))
        tk.Label(leg_frame, text="  ЛЕГЕНДА", bg=PANEL, fg=DIM,
            font=('Consolas', 8, 'bold')).pack(anchor='w', padx=8, pady=(6, 2))
        Legend(leg_frame).pack(fill='x', padx=4, pady=(0, 6))

        # Частотний діапазон + кнопки
        self._build_freq_panel(right)

        # Графіки
        plot_frame = tk.Frame(right, bg=BG)
        plot_frame.pack(fill='both', expand=True, pady=(4, 0))
        self._build_plots(plot_frame)

        # Панель інструментів — canvas вже існує
        self._build_toolbar()
        self._set_mode('select')

    # ── Панель інструментів ────────────────────────────────────────────────────

    def _build_toolbar(self):
        tb = tk.Frame(self, bg=HDR, height=46)
        tb.pack(fill='x');  tb.pack_propagate(False)

        # ─ Режим
        tk.Label(tb, text="Режим:", bg=HDR, fg=DIM,
            font=('Consolas', 9)).pack(side='left', padx=(12, 4), pady=10)

        self._mode_btns: dict = {}
        for lbl, mode in [("✦  Вибір", 'select'),
                          ("╌  Кабель", 'add_wire'),
                          ("✕  Видалити", 'delete')]:
            b = tk.Button(tb, text=lbl,
                bg=ENTRY, fg=TEXT, activebackground=BORD, activeforeground=TEXT,
                font=('Consolas', 9), relief='flat', bd=0,
                padx=10, pady=7, cursor='hand2',
                command=lambda m=mode: self._set_mode(m))
            b.pack(side='left', padx=2, pady=6)
            self._mode_btns[mode] = b

        # ─ Розділювач
        tk.Frame(tb, bg=BORD, width=1).pack(side='left', fill='y', padx=8, pady=6)
        tk.Label(tb, text="Додати:", bg=HDR, fg=DIM,
            font=('Consolas', 9)).pack(side='left', padx=(0, 4))

        self._node_btns: dict = {}
        for lbl, nt, tip in [
            ("📡 TX",   'plc_tx', "PLC Передавач"),
            ("📺 RX",   'plc_rx', "PLC Приймач"),
            ("⊥ Load",  'load',   "Навантаження"),
            ("●  Node", 'node',   "Вузол/Розгалуження"),
        ]:
            col = NODE_CLR[nt]
            b   = tk.Button(tb, text=lbl,
                bg=col, fg='white', activebackground=col, activeforeground='white',
                font=('Consolas', 9, 'bold'), relief='flat', bd=0,
                padx=8, pady=7, cursor='hand2',
                command=lambda n=nt: self._add_node_mode(n))
            b.pack(side='left', padx=2, pady=6)
            self._node_btns[nt] = b

        # ─ Дії
        tk.Frame(tb, bg=BORD, width=1).pack(side='left', fill='y', padx=8, pady=6)
        for lbl, bg, cmd in [
            ("⊡ Приклад",     ENTRY,  self._load_example),
            ("🗑 Очистити",    ENTRY,  self._clear_all),
        ]:
            tk.Button(tb, text=lbl,
                bg=bg, fg=DIM, activebackground=BORD, activeforeground=TEXT,
                font=('Consolas', 9), relief='flat', bd=0,
                padx=8, pady=7, cursor='hand2',
                command=cmd).pack(side='left', padx=2, pady=6)

        # Статус рядок
        self._status = tk.Label(tb, text="", bg=HDR, fg=GRN,
            font=('Consolas', 9))
        self._status.pack(side='right', padx=12)


    # ── Панель частот ─────────────────────────────────────────────────────────

    def _build_freq_panel(self, parent):
        fr = tk.Frame(parent, bg=PANEL)
        fr.pack(fill='x', pady=(4, 0))
        tk.Label(fr, text="  ЧАСТОТНИЙ ДІАПАЗОН", bg=PANEL, fg=ACC,
            font=('Consolas', 9, 'bold')).pack(anchor='w', padx=8, pady=(8, 4))
        for lbl, var in [("f мін [МГц]:", self._fmin),
                         ("f макс [МГц]:", self._fmax),
                         ("Точок:", self._fpts)]:
            row = tk.Frame(fr, bg=PANEL);  row.pack(fill='x', padx=10, pady=1)
            tk.Label(row, text=lbl, bg=PANEL, fg=DIM,
                font=('Consolas', 9), width=16, anchor='w').pack(side='left')
            tk.Entry(row, textvariable=var, width=8,
                bg=ENTRY, fg=TEXT, insertbackground=TEXT,
                relief='flat', font=('Consolas', 9)).pack(side='left', padx=4)

        bf = tk.Frame(fr, bg=PANEL);  bf.pack(fill='x', padx=10, pady=(6, 8))
        tk.Button(bf, text="▶  РОЗРАХУВАТИ",
            bg=BTN, fg='white', activebackground='#2EA043', activeforeground='white',
            font=('Consolas', 10, 'bold'), relief='flat', bd=0,
            padx=10, pady=7, cursor='hand2',
            command=self._calculate).pack(side='left', fill='x', expand=True)
        for sym, cmd in [("↺", self._clear_plots), ("💾", self._save_png)]:
            tk.Button(bf, text=sym,
                bg=ENTRY, fg=DIM, activebackground=BORD, activeforeground=TEXT,
                font=('Consolas', 10), relief='flat', bd=0,
                padx=8, pady=7, cursor='hand2',
                command=cmd).pack(side='left', padx=(4, 0))

    # ── Графіки ───────────────────────────────────────────────────────────────

    def _build_plots(self, parent):
        self.fig  = Figure(figsize=(5, 5), facecolor=PLT['bg'])
        self.ax_a = self.fig.add_subplot(211)
        self.ax_p = self.fig.add_subplot(212)
        self._style_axes()
        self.fig.tight_layout(pad=2.5)
        self.cv = FigureCanvasTkAgg(self.fig, master=parent)
        self.cv.draw()
        self.cv.get_tk_widget().pack(fill='both', expand=True)

    def _style_axes(self):
        for ax, title, yl in [
            (self.ax_a, "Загасання МБЕ", "Загасання [дБ]"),
            (self.ax_p, "Фаза МБЕ",       "Фаза [рад]"),
        ]:
            ax.set_facecolor(PLT['axes'])
            ax.tick_params(colors=PLT['text'], labelsize=7)
            for lb in (ax.xaxis.label, ax.yaxis.label, ax.title):
                lb.set_color(PLT['text'])
            for sp in ax.spines.values():
                sp.set_color(PLT['grid'])
            ax.grid(True, color=PLT['grid'], linewidth=0.5, linestyle='--')
            ax.set_title(title, fontsize=9, pad=4)
            ax.set_xlabel("Частота [МГц]", fontsize=8)
            ax.set_ylabel(yl, fontsize=8)

    # ── Логіка подій ──────────────────────────────────────────────────────────

    def _set_mode(self, mode):
        self.net_canvas.set_mode(mode)
        for m, b in self._mode_btns.items():
            b.configure(bg=ACC if m == mode else ENTRY,
                       fg=BG if m == mode else TEXT)
        for b in self._node_btns.values():
            b.configure(relief='flat')
        hints = {
            'select':   "Клік — вибір;  Перетягуй — переміщення;  Del — видалення",
            'add_wire': "Клікни на ПЕРШИЙ вузол, потім на ДРУГИЙ — з'єднає кабелем",
            'delete':   "Клікни на вузол або кабель щоб видалити",
        }
        self._status.configure(text=hints.get(mode, ''), fg=DIM)

    def _add_node_mode(self, ntype):
        self.net_canvas.set_mode('add_node', ntype)
        for m, b in self._mode_btns.items():
            b.configure(bg=ENTRY, fg=TEXT)
        for nt, b in self._node_btns.items():
            b.configure(relief='sunken' if nt == ntype else 'flat')
        names = {'plc_tx': 'PLC TX', 'plc_rx': 'PLC RX',
                 'load': 'Навантаження', 'node': 'Вузол'}
        self._status.configure(
            text=f"Клікніть на схемі щоб розмістити  [{names[ntype]}]", fg=GRN)

    def _on_select(self, etype, eid):
        self.props.show(etype, eid)
        if etype is not None:
            # Повертаємось у режим вибору після розміщення вузла
            if self.net_canvas.mode == 'add_node':
                pass   # Залишаємось в режимі — можна розмістити ще
            self.net_canvas.redraw()

    def _calculate(self):
        ok, msg = self.graph.validate()
        if not ok:
            messagebox.showerror("Помилка топології мережі", msg)
            return
        try:
            freq = np.linspace(
                self._fmin.get() * 1e6,
                self._fmax.get() * 1e6,
                max(10, int(self._fpts.get())))
            att, phi, pnodes, pedges = self.graph.build_cascade(freq)
            self._draw_plots(freq / 1e6, att, phi)
            self.net_canvas.highlight_path(pnodes, pedges)

            # Інформаційне повідомлення
            n_seg = len(pedges)
            n_br  = sum(1 for n, d in self.graph.nodes.items()
                        if d['type'] == 'load')
            jcts  = sum(1 for n in pnodes[1:-1]
                        if len([e for e in self.graph.adj[n]
                                if e not in set(pedges)]) > 0)
            self._status.configure(
                text=f"✓  Шлях: {n_seg} сегм.  │  "
                     f"Відгалуження: {jcts}  │  "
                     f"Навантаження: {n_br}  │  "
                     f"Загасання: {att.min():.1f}…{att.max():.1f} дБ",
                fg=GRN)
        except Exception as ex:
            messagebox.showerror("Помилка розрахунку", str(ex))

    def _draw_plots(self, freq_mhz, att, phi):
        col = PALETTE[self._plot_n % len(PALETTE)]
        lbl = (f"#{self._plot_n + 1}  "
               f"[{att.min():.1f}…{att.max():.1f} дБ]")
        self._plot_n += 1
        self.ax_a.plot(freq_mhz, att, color=col, linewidth=1.6, label=lbl)
        self.ax_p.plot(freq_mhz, phi, color=col, linewidth=1.6, label=lbl)
        for ax in (self.ax_a, self.ax_p):
            ax.legend(fontsize=7, facecolor=PLT['axes'],
                edgecolor=PLT['grid'], labelcolor=PLT['text'])
        self._style_axes()
        self.fig.tight_layout(pad=2.0)
        self.cv.draw()

    def _clear_plots(self):
        self.ax_a.cla();  self.ax_p.cla()
        self._style_axes()
        self.fig.tight_layout(pad=2.0)
        self.cv.draw()
        self._plot_n = 0

    def _save_png(self):
        p = filedialog.asksaveasfilename(
            defaultextension='.png', filetypes=[('PNG', '*.png')])
        if p:
            self.fig.savefig(p, dpi=150, facecolor=PLT['bg'], bbox_inches='tight')
            self._status.configure(text=f"💾  Збережено: {p}", fg=GRN)

    def _clear_all(self):
        if messagebox.askyesno("Очистити схему", "Видалити всю схему?"):
            self.graph = NetworkGraph()
            self.net_canvas.graph    = self.graph
            self.props.graph         = self.graph
            self.net_canvas.selected = None
            self.props.show(None, None)
            self.net_canvas.clear_path()
            self.net_canvas.redraw()
            self._set_mode('select')

    # ── Приклад мережі ────────────────────────────────────────────────────────

    def _load_example(self):
        """
        Розгалужена мережа:
        TX ─L1─▶ N1 ─L2─▶ N2 ─L3─▶ RX
                  │         │
                 L4        L5
                  ▼         ▼
                Load1      Load2
        """
        g = NetworkGraph()

        tx  = g.add_node('plc_tx', 120,  280, Z=75,  name='PLC-TX')
        n1  = g.add_node('node',   320,  280,         name='Вузол-1')
        n2  = g.add_node('node',   520,  280,         name='Вузол-2')
        rx  = g.add_node('plc_rx', 720,  280, Z=75,  name='PLC-RX')
        ld1 = g.add_node('load',   320,  460, Z=75,  name='Навант-1')
        ld2 = g.add_node('load',   520,  460, Z=150, name='Навант-2')

        g.add_edge(tx,  n1,  L=30, alpha=0.04, beta=0.18, Z0=150)
        g.add_edge(n1,  n2,  L=20, alpha=0.04, beta=0.18, Z0=150)
        g.add_edge(n2,  rx,  L=25, alpha=0.04, beta=0.18, Z0=150)
        g.add_edge(n1,  ld1, L=10, alpha=0.06, beta=0.20, Z0=150)
        g.add_edge(n2,  ld2, L=15, alpha=0.05, beta=0.19, Z0=150)

        self.graph = g
        self.net_canvas.graph    = g
        self.props.graph         = g
        self.net_canvas.selected = None
        self.props.show(None, None)
        self.net_canvas.clear_path()
        self.net_canvas.redraw()
        self._set_mode('select')
        self._status.configure(
            text="Приклад завантажено  —  натисніть ▶ РОЗРАХУВАТИ", fg=GRN)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app = MBENetApp()
    app.mainloop()