#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""贵州茅台日K走势 · 电影感动画（1920x1080 / 30fps / mp4）

风格：深色星云背景 + 发光折线 + 红绿光幕 + 事件信息卡 + 迷你时间轴 + 片头片尾。

依赖：pip install numpy pandas opencv-python pillow akshare    （另需 ffmpeg 在 PATH 中）

常用：
  python moutai_cinematic.py auto                      # 在线取数（akshare），输出 moutai_cinematic_2026.mp4
  python moutai_cinematic.py kline_data.csv            # 用本地 CSV（首次运行会自动保存一份）
  python moutai_cinematic.py auto --info               # 只打印时间线，不渲染
  python moutai_cinematic.py auto --preview 20         # 只导出第 20 秒的一帧 PNG，用来看版式
  python moutai_cinematic.py auto --audio bgm.mp3      # 加背景音乐（自动淡入淡出）
  python moutai_cinematic.py auto --size 1280x720      # 小尺寸，渲染更快

CSV 列（中英文表头均可）：日期/Date 开盘/Open 收盘/Close 最高/High 最低/Low 成交量/Volume
"""
import os

# 必须在导入 numpy 之前：限制每个进程的 BLAS 线程数，避免多进程时 OpenBLAS 抢光内存
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080

RED = (255, 98, 66)       # 上涨（A股红涨）
GREEN = (44, 228, 176)    # 下跌（A股绿跌）
GOLD = (238, 196, 128)
WHITE = (255, 255, 255)
GRAY = (168, 178, 190)
NAVY = (10, 24, 34)


def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def lerp(a, b, t):
    return a + (b - a) * t


def mix(c1, c2, t):
    return tuple(lerp(c1[i], c2[i], t) for i in range(3))


def rgba(c, a=1.0):
    return (int(c[0]), int(c[1]), int(c[2]), int(255 * clamp(a)))


# ----------------------------------------------------------------------------
# 字体
# ----------------------------------------------------------------------------
LINUX_NOTO = "/usr/share/fonts/opentype/noto/"
WIN = "C:/Windows/Fonts/"
FONT_CANDIDATES = {
    # 标题宋体（粗）
    "serif": [LINUX_NOTO + "NotoSerifCJK-Bold.ttc", WIN + "STZHONGS.TTF", WIN + "simsun.ttc",
              "/System/Library/Fonts/Supplemental/Songti.ttc", "/System/Library/Fonts/STHeiti Medium.ttc"],
    # 细体数字
    "num": [WIN + "segoeuil.ttf", WIN + "msyhl.ttc", LINUX_NOTO + "NotoSansCJK-Light.ttc",
            "/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    # 常规黑体
    "sans": [WIN + "msyh.ttc", LINUX_NOTO + "NotoSansCJK-Regular.ttc", WIN + "simhei.ttf",
             "/System/Library/Fonts/PingFang.ttc", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"],
    # 粗黑体
    "sansb": [WIN + "msyhbd.ttc", LINUX_NOTO + "NotoSansCJK-Bold.ttc", WIN + "simhei.ttf",
              "/System/Library/Fonts/PingFang.ttc", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"],
}


def _open_font(path, size):
    if not path or not os.path.exists(path):
        return None
    try:
        if path.lower().endswith((".ttc", ".otc")):
            first = None
            for idx in range(8):
                try:
                    f = ImageFont.truetype(path, size, index=idx)
                except Exception:
                    break
                if first is None:
                    first = f
                if "SC" in " ".join(f.getname()):
                    return f
            return first
        return ImageFont.truetype(path, size)
    except Exception:
        return None


class Fonts:
    def __init__(self, overrides):
        self.over = overrides or {}
        self.cache = {}
        self.native_bold = {}

    def get(self, kind, size):
        key = (kind, size)
        if key in self.cache:
            return self.cache[key]
        paths = ([self.over[kind]] if self.over.get(kind) else []) + FONT_CANDIDATES[kind]
        f = None
        for p in paths:
            f = _open_font(p, size)
            if f is not None:
                self.native_bold[kind] = ("bold" in os.path.basename(p).lower()
                                          or "bd" in os.path.basename(p).lower())
                break
        if f is None:  # 最后兜底：用 matplotlib 找一个中文字体
            try:
                from matplotlib import font_manager as fm
                for fam in ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Micro Hei"]:
                    try:
                        f = ImageFont.truetype(fm.findfont(fam, fallback_to_default=False), size)
                        break
                    except Exception:
                        continue
            except Exception:
                pass
        if f is None:
            f = ImageFont.load_default()
        self.cache[key] = f
        return f


def text_w(font, s, tracking=0):
    if tracking == 0:
        return font.getlength(s)
    return sum(font.getlength(ch) for ch in s) + tracking * max(len(s) - 1, 0)


def draw_text(d, xy, s, font, fill, anchor="l", tracking=0, stroke=0):
    """以基线定位；anchor: l / m / r"""
    x, y = xy
    w = text_w(font, s, tracking)
    if anchor == "r":
        x -= w
    elif anchor == "m":
        x -= w / 2
    if tracking == 0:
        d.text((x, y), s, font=font, fill=fill, anchor="ls", stroke_width=stroke, stroke_fill=fill)
    else:
        for ch in s:
            d.text((x, y), ch, font=font, fill=fill, anchor="ls", stroke_width=stroke, stroke_fill=fill)
            x += font.getlength(ch) + tracking
    return w


def new_layer(w, h, base=(255, 255, 255)):
    """透明图层；RGB 预填，避免文字抗锯齿边缘出现黑边"""
    return Image.new("RGBA", (w, h), (base[0], base[1], base[2], 0))


def blit(out, layer, alpha):
    """layer = (y0, rgba uint8 array)；alpha 混合到 float 画面"""
    if alpha <= 0.001:
        return
    y0, arr = layer
    h = arr.shape[0]
    a = arr[..., 3:4].astype(np.float32) * (alpha / 255.0)
    out[y0:y0 + h] = out[y0:y0 + h] * (1 - a) + arr[..., :3].astype(np.float32) * a


# ----------------------------------------------------------------------------
# 渲染器
# ----------------------------------------------------------------------------
class Renderer:
    def __init__(self, a):
        self.a = a
        cv2.setNumThreads(1)
        self.fonts = Fonts({"serif": a.font_serif, "sans": a.font_sans, "num": a.font_num})
        self.load_data()
        self.prep_bg()
        self.prep_title()

    # ---------------- 数据 ----------------
    def load_data(self):
        a = self.a
        df = pd.read_csv(a.data_csv, parse_dates=["Date"])
        self.df = df
        self.N = N = len(df)
        self.dates = list(df["Date"])
        self.o = df["Open"].to_numpy(float)
        self.c = c = df["Close"].to_numpy(float)
        self.hi_arr = df["High"].to_numpy(float) if "High" in df else c
        self.lo_arr = df["Low"].to_numpy(float) if "Low" in df else c
        # 折线头部在开/高/低/收之间快速跳动（需要 High/Low 列）
        self.zz = bool(a.jitter and "High" in df and "Low" in df)
        self.r = df["Ret"].to_numpy(float)
        self.runmax = np.maximum.accumulate(c)
        self.dd = c / self.runmax - 1
        self.trend = pd.Series(self.r).rolling(5, min_periods=1).mean().to_numpy()
        ret20 = pd.Series(c).pct_change(20).fillna(0).to_numpy()
        self.warm = np.clip(0.35 + 6 * ret20, 0.0, 1.0)

        # 相机：y 轴范围随已播放数据变化（平滑、带少量前瞻）
        def sm(x, k=9):
            pad = np.pad(x, (k // 2, k // 2), mode="edge")
            return np.convolve(pad, np.ones(k) / k, mode="valid")
        lo_src = self.lo_arr if self.zz else c
        hi_src = self.hi_arr if self.zz else c
        # 滚动模式：当前点固定在画面中央，纵轴只跟随画面里可见的这段（往前约 W/2/px 天）
        self.scroll = not a.fit
        if self.scroll:
            wd = int(math.ceil((W / 2) / a.px)) + 4
            cm = pd.Series(lo_src).rolling(wd, min_periods=1).min().to_numpy()
            cM = pd.Series(hi_src).rolling(wd, min_periods=1).max().to_numpy()
            ksm = 15
        else:
            cm, cM, ksm = np.minimum.accumulate(lo_src), np.maximum.accumulate(hi_src), 9
        lo, hi = np.minimum(sm(cm, ksm), cm), np.maximum(sm(cM, ksm), cM)
        span = np.maximum(hi - lo, 0.05 * c)
        mid = (hi + lo) / 2
        lo, hi = mid - span / 2, mid + span / 2
        pad = (hi - lo) * 0.08
        self.cam_lo, self.cam_hi = lo - pad, hi + pad

        # 事件：落到不早于事件日期的第一个交易日
        evs = []
        if a.events and os.path.exists(a.events):
            with open(a.events, encoding="utf-8") as f:
                raw = json.load(f)
            for e in raw:
                d0 = pd.Timestamp(e["date"])
                hit = np.where(df["Date"].to_numpy() >= np.datetime64(d0))[0]
                if len(hit) == 0:
                    continue
                lines = e.get("lines")
                if not lines:
                    lines = [s for s in str(e.get("detail", "")).replace("；", "\n").split("\n") if s]
                evs.append(dict(idx=int(hit[0]), date=d0, tag=e.get("tag", "事件"),
                                title=e["title"], lines=lines[:2]))
        evs.sort(key=lambda e: e["idx"])
        self.evs = evs

        # 月份、季度
        self.month_pos = [i for i in range(N) if i == 0 or self.dates[i].month != self.dates[i - 1].month]
        self.qstart = {}
        for i, d in enumerate(self.dates):
            self.qstart.setdefault((d.year, (d.month - 1) // 3), i)

        self.gmin, self.gmax = c.min(), c.max()
        self.colnoise = np.random.default_rng(3).uniform(0.72, 1.0, W + 4).astype(np.float32)

    # ---------------- 背景 ----------------
    def prep_bg(self):
        rng = np.random.default_rng(7)

        def layer(sx, sy):
            arr = rng.random((sy, sx)).astype(np.float32)
            return cv2.resize(arr, (W + 200, H), interpolation=cv2.INTER_CUBIC)
        neb = 0.55 * layer(6, 4) + 0.3 * layer(13, 8) + 0.15 * layer(26, 15)
        neb = cv2.GaussianBlur(neb, (0, 0), 30)
        neb = (neb - neb.min()) / (neb.max() - neb.min())
        neb = np.clip((neb - 0.3) / 0.7, 0, 1) ** 1.6
        # 光集中在画面中下部
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        neb *= (0.35 + 0.65 * np.exp(-((yy - 0.62) ** 2) / 0.12))
        self.neb = neb.astype(np.float32)
        xx = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
        yv = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
        self.vig = np.clip(1.0 - 0.5 * (xx ** 2 * 0.55 + yv ** 2 * 0.9), 0.25, 1).astype(np.float32)

        n = 170
        self.pt = dict(
            x=rng.uniform(0, W, n), y=rng.uniform(0, H, n),
            vx=rng.uniform(-3, 6, n), vy=rng.uniform(-14, -2, n),
            ph=rng.uniform(0, 6.28, n), sp=rng.uniform(0.6, 2.2, n),
            br=rng.uniform(0.25, 1.0, n), big=rng.random(n) < 0.18,
            gold=rng.random(n) < 0.55)

    def bg(self, t, warm):
        off = int((t * 0.22) % 200)
        neb = self.neb[:, off:off + W]
        teal = np.array([0.03, 0.30, 0.36], np.float32)
        hot = np.array([0.38, 0.14, 0.08], np.float32)
        tint = (1 - warm) * teal + warm * hot
        out = neb[..., None] * tint[None, None, :] * 255.0 * 0.85
        out += np.array([3, 5, 9], np.float32)
        out *= self.vig[..., None]
        # 粒子
        P = self.pt
        ts = t / self.a.fps
        x = ((P["x"] + P["vx"] * ts) % W).astype(int)
        y = ((P["y"] + P["vy"] * ts) % H).astype(int)
        tw = 0.5 + 0.5 * np.sin(P["ph"] + ts * P["sp"])
        b = (P["br"] * tw * 255).astype(np.float32)
        col = np.where(P["gold"][:, None], np.array([255, 214, 150.0]), np.array([150, 230, 220.0])) / 255.0
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            m = P["big"] if (dx or dy) else np.ones_like(P["big"])
            yy_, xx_ = np.clip(y + dy, 0, H - 1)[m], np.clip(x + dx, 0, W - 1)[m]
            out[yy_, xx_] += col[m] * b[m][:, None]
        return out

    # ---------------- 片头 ----------------
    def _strip(self, y0, h, fn, base=WHITE):
        im = new_layer(W, h, base)
        d = ImageDraw.Draw(im)
        fn(d)
        return (y0, np.asarray(im).copy())

    def prep_title(self):
        a, F = self.a, self.fonts
        d0, d1 = self.dates[0], self.dates[-1]
        c0, c1 = self.c[0], self.c[-1]
        cx = W // 2

        def eyebrow(d):
            draw_text(d, (cx, 50), f"{a.name_en} · {a.symbol} · {d0:%Y.%m.%d} — {d1:%Y.%m.%d}",
                      F.get("sans", 19), rgba(GRAY, 0.85), "m", tracking=8)
        self.t_eyebrow = self._strip(290, 70, eyebrow, GRAY)

        # 金色渐变大标题 + 辉光
        size = 156
        font = F.get("serif", size)
        stroke = 0 if F.native_bold.get("serif") else 3
        h_strip = 300
        mask = Image.new("L", (W, h_strip), 0)
        md = ImageDraw.Draw(mask)
        draw_text(md, (cx, 218), a.name, font, 255, "m", tracking=30, stroke=stroke)
        m = np.asarray(mask).astype(np.float32) / 255.0
        grad = np.linspace(0, 1, h_strip, dtype=np.float32)[:, None]
        top, bot = np.array([255, 240, 205], np.float32), np.array([214, 168, 100], np.float32)
        gr = np.clip((grad - 0.25) / 0.45, 0, 1)
        colr = top[None, None, :] * (1 - gr[..., None]) + bot[None, None, :] * gr[..., None]
        colr = np.broadcast_to(colr, (h_strip, W, 3))
        arr = np.dstack([colr, m[..., None] * 255.0]).astype(np.uint8)
        self.t_title = (330, arr)
        glow = cv2.GaussianBlur(colr * m[..., None], (0, 0), 22) * 1.1
        self.t_title_glow = (330, glow.astype(np.float32))

        # 细线
        rule = np.zeros((20, W, 4), np.uint8)
        xs = np.arange(W)
        prof = np.exp(-((xs - cx) / 150.0) ** 4)
        rule[9:11, :, :3] = (214, 190, 150)
        rule[9:11, :, 3] = (prof * 150).astype(np.uint8)[None, :]
        self.t_rule = (590, rule)

        # 副标题（金色数字）
        def sub(d):
            fs, fn = F.get("sans", 40), F.get("num", 46)
            runs = [(f"{d0.year} 年至今，从 ", fs, GRAY), (f"{c0:,.0f}", fn, GOLD), (" 元到 ", fs, GRAY),
                    (f"{c1:,.0f}", fn, GOLD), (" 元", fs, GRAY)]
            tr = 6
            total = sum(text_w(f, s, tr) + tr for s, f, _ in runs)
            x = cx - total / 2
            for s, f, col in runs:
                x += draw_text(d, (x, 58), s, f, rgba(col, 0.95), tracking=tr) + tr
        self.t_sub = self._strip(640, 90, sub, GRAY)

    def frame_title(self, idx, spec):
        a = self.a
        T = a.title_sec
        ts = idx / a.fps
        g = 1 - smooth((ts - (T - 0.9)) / 0.8)
        out = self.bg(idx, 0.35)
        blit(out, self.t_eyebrow, smooth((ts - 0.2) / 0.9) * g)
        at = smooth((ts - 0.5) / 1.1) * g
        y0, gl = self.t_title_glow
        out[y0:y0 + gl.shape[0]] += gl * (at * 0.8)
        blit(out, self.t_title, at)
        blit(out, self.t_rule, smooth((ts - 1.0) / 0.8) * g)
        blit(out, self.t_sub, smooth((ts - 1.5) / 1.0) * g)
        return np.clip(out, 0, 255).astype(np.uint8)

    # ---------------- 图表核心 ----------------
    def cam(self, p):
        x = np.arange(self.N)
        return float(np.interp(p, x, self.cam_lo)), float(np.interp(p, x, self.cam_hi))

    def main_view(self, p):
        lo, hi = self.cam(p)
        if self.scroll:   # 当前时间点固定在水平中央，K线随时间向左滚动
            xr = W / 2.0
            return dict(xl=xr - self.a.px * p, xr=xr, ytop=380.0, ybot=760.0, lo=lo, hi=hi)
        XL, HXMAX, SMAX = 96.0, 1240.0, 16.0
        s = min(SMAX, (HXMAX - XL) / max(p, 1.0))
        return dict(xl=XL, xr=XL + s * max(p, 1e-6), ytop=380.0, ybot=760.0, lo=lo, hi=hi)

    def end_view(self, e):
        lo, hi = self.cam(self.N - 1)
        mv = self.main_view(self.N - 1)
        glo, ghi = self.gmin, self.gmax
        pad = (ghi - glo) * 0.08
        return dict(xl=lerp(mv["xl"], 210.0, e), xr=lerp(mv["xr"], 1710.0, e),
                    ytop=lerp(mv["ytop"], 470.0, e), ybot=lerp(mv["ybot"], 870.0, e),
                    lo=lerp(lo, glo - pad, e), hi=lerp(hi, ghi + pad, e))

    @staticmethod
    def vy(v, view):
        return view["ybot"] - (v - view["lo"]) / (view["hi"] - view["lo"]) * (view["ybot"] - view["ytop"])

    def point(self, i, p, view):
        x = view["xl"] + (view["xr"] - view["xl"]) * (i / max(p, 1e-6))
        return x, self.vy(self.c[int(i)], view)

    def head_val(self, k, j, f):
        """折线头部的价格：从昨收出发，沿 开→低/高→高/低→收 的路径快速上下跳动，f 为当日进度 0~1"""
        if not self.zz:
            return float(lerp(self.c[k], self.c[j], f))
        o, h, l, cc = self.o[j], self.hi_arr[j], self.lo_arr[j], self.c[j]
        x1, x2 = (l, h) if cc >= o else (h, l)
        return float(np.interp(f, [0, 0.2, 0.45, 0.7, 1.0], [self.c[k], o, x1, x2, cc]))

    def efade(self, x, edge):
        """滚动模式下，折线在画面左边缘渐隐；edge=1 全开，0 关闭（片尾拉远时逐渐关掉）"""
        if not self.scroll or edge <= 0:
            return 1.0 + 0 * np.asarray(x, dtype=float)
        f = np.clip(np.asarray(x, dtype=float) / 260.0, 0, 1)
        f = f * f * (3 - 2 * f)
        return 1 - edge * (1 - f)

    def chart(self, t, p, view, warm, marks=(), ring=None, grid=True, edge=0.0, now_line=False):
        """返回 (uint8 画面, 几何信息)。marks: [(x,y,color,radius,alpha)]；ring: (x,y,color,alpha,t)"""
        N, c = self.N, self.c
        out = self.bg(t, warm)

        k = int(math.floor(p))
        idxs = np.arange(k + 1, dtype=np.float64)
        vals = c[:k + 1]
        if p - k > 1e-6 and k + 1 < N:
            idxs = np.append(idxs, p)
            vals = np.append(vals, self.head_val(k, k + 1, p - k))
        if p < 1e-6:
            xs = np.array([view["xl"]])
        else:
            xs = view["xl"] + (view["xr"] - view["xl"]) * (idxs / p)
        ys = self.vy(vals, view)
        if len(xs) > 2:   # 只保留画面内（及左边缘外一点点）的顶点
            keep = np.where(xs >= -80)[0]
            s0 = max(0, int(keep[0]) - 1) if len(keep) else len(xs) - 2
            xs, ys, idxs = xs[s0:], ys[s0:], idxs[s0:]
        head = (float(xs[-1]), float(ys[-1]))

        # 水平虚线（整数价位）
        if grid:
            span = view["hi"] - view["lo"]
            step = next(s for s in (10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000) if span / s <= 5)
            first = math.ceil(view["lo"] / step) * step
            gm = np.zeros((H, W), np.uint8)
            levels = []
            v = first
            while v < view["hi"]:
                y = int(self.vy(v, view))
                if 200 < y < 940:
                    for x in range(0, W, 30):
                        cv2.line(gm, (x, y), (x + 16, y), 255, 1)
                    levels.append((v, y))
                v += step
            ticks = []
            if self.scroll and p > 1e-6:   # 月份竖向刻度，随图表一起向左滚动（右侧是尚未到来的月份）
                for i in self.month_pos:
                    x = view["xl"] + (view["xr"] - view["xl"]) * (i / p)
                    if 0 < x < W:
                        for y in range(330, 945, 26):
                            cv2.line(gm, (int(x), y), (int(x), y + 12), 255, 1)
                        ticks.append((x, f"{self.dates[i].year}" if i == 0 else f"{self.dates[i].month}月"))
            out += (gm[..., None].astype(np.float32) / 255.0) * np.array([120, 145, 155], np.float32) * 0.2
        else:
            levels, ticks = [], []
        if now_line:   # 画面中央的“此刻”竖线
            nx = int(view["xr"])
            prof = np.clip(1 - np.abs(np.linspace(-1, 1, 700)) ** 3, 0, 1).astype(np.float32)
            out[300:1000, nx - 1:nx + 1] += prof[:, None, None] * np.array([120, 150, 160], np.float32) * 0.45

        # 红绿光幕
        if len(xs) >= 2:
            cols = np.arange(max(0, int(math.ceil(xs[0]))), int(xs[-1]) + 1)
            if len(cols) >= 2:
                pos = np.interp(cols, xs, idxs)
                ly = np.interp(cols, xs, ys)
                di = np.clip(np.floor(pos + 0.5).astype(int), 0, N - 1)
                rr = self.r[di]
                big = np.clip(np.abs(rr) / 0.03, 0, 1)
                colv = np.where((rr >= 0)[:, None], np.array(RED, np.float32), np.array(GREEN, np.float32))
                strength = (0.13 + 0.55 * big) * self.colnoise[cols] * self.efade(cols, edge)
                L = 150 + 520 * big
                r0 = int(max(0, ly.min() - 10))
                r1 = 1000
                Y = np.arange(r0, r1, dtype=np.float32)[:, None]
                dy = Y - ly[None, :].astype(np.float32)
                al = np.exp(-np.maximum(dy, 0) / L[None, :]) * (dy >= 0) * strength[None, :]
                al *= np.clip((968 - Y) / 180.0, 0, 1)
                out[r0:r1, cols[0]:cols[-1] + 1] += al[..., None] * colv[None, :, :]

        # 发光折线
        lay = np.zeros((H, W, 3), np.uint8)
        sh = 4
        sc = 1 << sh

        def P_(x, y):
            return (int(round(x * sc)), int(round(y * sc)))
        for j in range(len(xs) - 1):
            di = int(min(idxs[j + 1], N - 1))
            tr = self.trend[di]
            s = clamp(abs(tr) / 0.010)
            col = mix((255, 244, 232), RED if tr >= 0 else GREEN, 0.25 + 0.75 * s)
            fd = float(self.efade((xs[j] + xs[j + 1]) / 2, edge))
            cv2.line(lay, P_(xs[j], ys[j]), P_(xs[j + 1], ys[j + 1]), tuple(int(v * fd) for v in col), 5, cv2.LINE_AA, sh)
        cv2.circle(lay, P_(*head), 7 * sc // sc, (255, 240, 220), -1, cv2.LINE_AA)
        cv2.line(lay, (int(head[0]) - 110, int(head[1])), (int(head[0]) + 110, int(head[1])), (255, 226, 190), 1, cv2.LINE_AA)
        marks = [(mx, my, mc, mr, ma * float(self.efade(mx, edge))) for (mx, my, mc, mr, ma) in marks
                 if -20 < mx < W + 20]
        for (mx, my, mc, mr, ma) in marks:
            cv2.circle(lay, (int(mx), int(my)), int(mr), tuple(int(v * ma) for v in mc), -1, cv2.LINE_AA)
        if ring is not None:
            rx, ry, rc, ra, rt = ring
            col = tuple(int(v * ra) for v in rc)
            rad = 34 + 3 * math.sin(rt * 0.18)
            cv2.circle(lay, (int(rx), int(ry)), int(rad), col, 2, cv2.LINE_AA)
            cv2.line(lay, (int(rx), int(ry) - 280), (int(rx), int(ry) + 280),
                     tuple(int(v * ra * 0.85) for v in rc), 2, cv2.LINE_AA)

        half = cv2.resize(lay, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        quarter = cv2.resize(half, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        g1 = cv2.resize(cv2.GaussianBlur(half, (0, 0), 4.5), (W, H), interpolation=cv2.INTER_LINEAR)
        g2 = cv2.resize(cv2.GaussianBlur(quarter, (0, 0), 7), (W, H), interpolation=cv2.INTER_LINEAR)
        out += g1.astype(np.float32) * 0.9 + g2.astype(np.float32) * 1.15 + lay.astype(np.float32) * 0.5
        img = np.clip(out, 0, 255).astype(np.uint8)

        # 白色线芯
        for j in range(len(xs) - 1):
            fd = float(self.efade((xs[j] + xs[j + 1]) / 2, edge))
            if fd > 0.04:
                cv2.line(img, P_(xs[j], ys[j]), P_(xs[j + 1], ys[j + 1]),
                         (int(255 * fd), int(252 * fd), int(246 * fd)), 2, cv2.LINE_AA, sh)
        cv2.circle(img, P_(*head), 4 * sc // sc, (255, 255, 255), -1, cv2.LINE_AA)
        for (mx, my, mc, mr, ma) in marks:
            cv2.circle(img, (int(mx), int(my)), 2, (255, 255, 255), -1, cv2.LINE_AA)
        return img, dict(xs=xs, ys=ys, idxs=idxs, head=head, levels=levels, ticks=ticks)

    # ---------------- 主画面 ----------------
    def frame_main(self, idx, spec):
        _, p, evk, hf, mf = spec
        a, F = self.a, self.fonts
        N = self.N
        warm = float(np.interp(p, np.arange(N), self.warm))
        view = self.main_view(p)
        ua = smooth(mf / 24.0)
        icur = int(min(N - 1, math.floor(p + 0.5)))

        marks = []
        for e in self.evs:
            if e["idx"] <= p - 1e-9 or (evk >= 0 and e is self.evs[evk]):
                if e["idx"] <= p + 1e-9:
                    x, y = self.point(e["idx"], p, view)
                    marks.append((x, y, RED if self.r[e["idx"]] >= 0 else GREEN, 5, 0.9))
        ring, card_a = None, 0.0
        if evk >= 0:
            card_a = smooth(min(hf / 10.0, (a.hold_frames - 1 - hf) / 10.0))
            e = self.evs[evk]
            x, y = self.point(e["idx"], p, view)
            ring = (x, y, RED if self.r[e["idx"]] >= 0 else GREEN, card_a, idx)
        img, geo = self.chart(idx, p, view, warm, marks, ring,
                              edge=1.0 if self.scroll else 0.0, now_line=self.scroll)

        ov = np.zeros((H, W, 3), np.uint8)
        self.draw_minimap(ov, p, evk)
        if evk >= 0 and card_a > 0.01:
            e = self.evs[evk]
            acc = RED if self.r[e["idx"]] >= 0 else GREEN
            hx, hy = geo["head"]
            cv2.line(ov, (int(hx), int(hy)), (1305, 342), tuple(int(v * 0.75 * card_a) for v in acc), 1, cv2.LINE_AA)
        img = cv2.add(img, cv2.convertScaleAbs(ov, alpha=ua) if ua < 0.999 else ov)

        ui = new_layer(W, H)
        d = ImageDraw.Draw(ui)
        self.draw_header(d, icur, p, ua, geo)
        self.draw_grid_labels(d, geo, ua)
        for x, lab in geo.get("ticks", []):
            ta = 0.55 * ua * float(self.efade(x, 1.0))
            if ta > 0.02:
                draw_text(d, (x, 936), lab, self.fonts.get("num", 19), rgba(GRAY, ta), "m", tracking=1)
        self.draw_months(d, ua)
        if evk >= 0 and card_a > 0.01:
            self.draw_card(d, self.evs[evk], card_a)
        return self.compose(img, ui)

    def compose(self, img, ui):
        base = Image.fromarray(img).convert("RGBA")
        return np.asarray(Image.alpha_composite(base, ui).convert("RGB"))

    def draw_header(self, d, i, p, ua, geo):
        F = self.fonts
        dt = self.dates[i]
        draw_text(d, (96, 87), f"{self.a.name_en} · {self.a.symbol}", F.get("sans", 19), rgba(GRAY, 0.8 * ua), tracking=5)
        draw_text(d, (96, 124), f"{self.a.name} · 日K走势", F.get("sansb", 23), rgba(WHITE, 0.92 * ua), tracking=3)
        draw_text(d, (92, 258), f"{dt.year}", F.get("num", 136), rgba(WHITE, 0.98 * ua), tracking=2)
        draw_text(d, (98, 308), f"{dt.month:02d} · {dt.day:02d}", F.get("num", 27), rgba(GRAY, 0.9 * ua), tracking=7)

        tr = self.trend[i]
        s = clamp(abs(tr) / 0.010)
        pc = mix(WHITE, RED if tr >= 0 else GREEN, 0.35 + 0.65 * s)
        draw_text(d, (1824, 92), "收盘价", F.get("sans", 19), rgba(GRAY, 0.8 * ua), "r", tracking=4)
        draw_text(d, (1822, 192), f"{self.c[i]:,.2f}", F.get("num", 96), rgba(pc, 0.98 * ua), "r", tracking=2)
        dd = self.dd[i]
        f = F.get("sans", 25)
        if dd > -1e-9:
            txt, col, up = "创年内新高", RED, True
        else:
            txt, col, up = f"距年内高点 {dd * 100:.1f}%", GREEN, False
        w = text_w(f, txt, 2)
        draw_text(d, (1824, 240), txt, f, rgba(col, 0.95 * ua), "r", tracking=2)
        tx, ty = 1824 - w - 26, 230
        tri = [(tx, ty - 14), (tx + 18, ty - 14), (tx + 9, ty + 2)] if not up else [(tx, ty + 2), (tx + 18, ty + 2), (tx + 9, ty - 14)]
        d.polygon(tri, fill=rgba(col, 0.95 * ua))

        # 季度标签
        q = (dt.month - 1) // 3
        qi = self.qstart.get((dt.year, q), 0)
        qa = smooth((p - qi) / 1.2) * ua
        names = ["第一季度", "第二季度", "第三季度", "第四季度"]
        draw_text(d, (98, 790), f"{dt.year}.{3 * q + 1:02d} — {dt.year}.{3 * q + 3:02d}", F.get("num", 25),
                  rgba(GOLD, 0.9 * qa), tracking=5)
        font = F.get("serif", 70)
        draw_text(d, (96, 872), names[q], font, rgba((250, 244, 232), 0.95 * qa), tracking=8,
                  stroke=0 if F.native_bold.get("serif") else 2)

    def draw_grid_labels(self, d, geo, ua):
        f = self.fonts.get("num", 19)
        for v, y in geo["levels"]:
            draw_text(d, (98, y - 9), f"{v:,.0f}", f, rgba(GRAY, 0.55 * ua), tracking=2)

    def draw_months(self, d, ua):
        f = self.fonts.get("num", 18)
        for i in self.month_pos:
            x = 96 + (1824 - 96) * i / max(self.N - 1, 1)
            lab = f"{self.dates[i].year}" if i == 0 else f"{self.dates[i].month}月"
            draw_text(d, (x, 1052), lab, f if i else f, rgba(GRAY, 0.6 * ua), "m", tracking=1)

    def draw_minimap(self, ov, p, evk):
        N = self.N
        x0, x1, y0, y1 = 96.0, 1824.0, 972.0, 1008.0
        cv2.line(ov, (96, 1024), (1824, 1024), (60, 70, 78), 1, cv2.LINE_AA)
        k = int(math.floor(p))
        xs = [x0 + (x1 - x0) * i / (N - 1) for i in range(k + 1)]
        vs = list(self.c[:k + 1])
        if p - k > 1e-6 and k + 1 < N:
            xs.append(x0 + (x1 - x0) * p / (N - 1))
            vs.append(float(np.interp(p, np.arange(N), self.c)))
        ys = [y1 - (v - self.gmin) / (self.gmax - self.gmin + 1e-9) * (y1 - y0) for v in vs]
        if len(xs) >= 2:
            pts = np.stack([xs, ys], 1).round().astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(ov, [pts], False, (205, 214, 224), 1, cv2.LINE_AA)
        for e in self.evs:
            if e["idx"] <= p + 1e-9:
                col = RED if self.r[e["idx"]] >= 0 else GREEN
                ex = x0 + (x1 - x0) * e["idx"] / (N - 1)
                ey = y1 - (self.c[e["idx"]] - self.gmin) / (self.gmax - self.gmin + 1e-9) * (y1 - y0)
                cv2.circle(ov, (int(ex), int(ey)), 3, col, -1, cv2.LINE_AA)
        cv2.circle(ov, (int(xs[-1]), int(ys[-1])), 7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(ov, (int(xs[-1]), int(ys[-1])), 3, (255, 190, 120), -1, cv2.LINE_AA)

    def draw_card(self, d, e, a):
        F = self.fonts
        acc = RED if self.r[e["idx"]] >= 0 else GREEN
        slide = (1 - a) * 28
        CX, CY, CW, CH = 1305 + slide, 292, 517, 293
        d.rectangle([CX, CY, CX + CW, CY + CH], fill=rgba(NAVY, 0.86 * a), outline=rgba(WHITE, 0.09 * a))
        d.rectangle([CX, CY, CX + 4, CY + CH], fill=rgba(acc, 0.95 * a))
        x = CX + 38
        wt = draw_text(d, (x, CY + 52), e["tag"], F.get("sansb", 21), rgba(acc, a), tracking=3)
        draw_text(d, (x + wt + 26, CY + 52), f"{e['date']:%Y.%m.%d}", F.get("num", 21), rgba(GRAY, 0.9 * a), tracking=2)
        draw_text(d, (x, CY + 118), e["title"], F.get("serif", 44), rgba(WHITE, a), tracking=2,
                  stroke=0 if F.native_bold.get("serif") else 1)
        r = self.r[e["idx"]] * 100
        wp = draw_text(d, (x, CY + 190), f"{r:+.2f}%", F.get("num", 64), rgba(acc, a), tracking=2)
        draw_text(d, (x + wp + 18, CY + 188), "当日涨跌", F.get("sans", 20), rgba(GRAY, 0.85 * a), tracking=2)
        for j, line in enumerate(e["lines"][:2]):
            draw_text(d, (x, CY + 238 + 34 * j), line, F.get("sans", 23), rgba((178, 188, 198), 0.95 * a), tracking=1)

    # ---------------- 片尾 ----------------
    def frame_end(self, idx, spec):
        _, u = spec
        a, F, N = self.a, self.fonts, self.N
        e = smooth(u / 1.6)
        view = self.end_view(e)
        p = N - 1
        warm = float(self.warm[-1])
        ua = 1 - smooth(u / 0.8)
        fa = smooth((u - 1.4) / 1.0)      # 统计文字
        ma = smooth((u - 1.8) / 1.0)      # 标注
        foot = smooth((u - 2.6) / 1.0)

        # 标注点
        hi_i, lo_i = int(np.argmax(self.c)), int(np.argmin(self.c))
        marks = []
        if ma > 0:
            for i, col in ((0, GOLD), (hi_i, RED), (lo_i, GREEN), (N - 1, GOLD)):
                x, y = self.point(i, p, view)
                marks.append((x, y, col, 6, ma))
            for ev in self.evs:
                x, y = self.point(ev["idx"], p, view)
                marks.append((x, y, RED if self.r[ev["idx"]] >= 0 else GREEN, 4, ma * 0.8))
        img, geo = self.chart(idx, p, view, warm, marks, None, grid=False,
                              edge=(1 - e) if self.scroll else 0.0)

        ov = np.zeros((H, W, 3), np.uint8)
        if ma > 0.01:   # 年内高点金色横线
            hx, hy = self.point(hi_i, p, view)
            lx, _ = self.point(N - 1, p, view)
            cv2.line(ov, (int(hx), int(hy)), (int(lx), int(hy)), tuple(int(v * 0.65 * ma) for v in GOLD), 1, cv2.LINE_AA)
        ev_rows = []
        if ma > 0.01:   # 事件：底部两行错开的标签，细线连到折线上的点
            for j, ev in enumerate(self.evs):
                x, y = self.point(ev["idx"], p, view)
                acc = RED if self.r[ev["idx"]] >= 0 else GREEN
                ty = 928 + 38 * (j % 2)
                cv2.line(ov, (int(x), int(y) + 12), (int(x), ty - 24),
                         tuple(int(v * 0.35 * ma) for v in acc), 1, cv2.LINE_AA)
                ev_rows.append((x, ty, ev["tag"], acc))
        if ua > 0.01:
            self.draw_minimap(ov, p, -1)
        img = cv2.add(img, ov)

        ui = new_layer(W, H)
        d = ImageDraw.Draw(ui)
        if ua > 0.01:
            self.draw_header(d, N - 1, p, ua, geo)
            self.draw_months(d, ua)

        # 统计块
        c0, c1 = self.c[0], self.c[-1]
        chg = (c1 / c0 - 1) * 100
        acc = RED if chg >= 0 else GREEN
        draw_text(d, (98, 152), f"{self.dates[0]:%Y.%m.%d} → {self.dates[-1]:%Y.%m.%d}", F.get("num", 24),
                  rgba(GRAY, 0.85 * fa), tracking=5)
        draw_text(d, (96, 238), f"{c0:,.2f} → {c1:,.2f}", F.get("num", 66), rgba((226, 232, 238), 0.95 * fa), tracking=2)
        sb = 0 if F.native_bold.get("serif") else 3
        wl = draw_text(d, (98, 362), "区间 ", F.get("serif", 60), rgba(GOLD, 0.95 * fa), tracking=6, stroke=sb)
        draw_text(d, (98 + wl + 12, 362), f"{chg:+.1f}%", F.get("serif", 118), rgba(mix(GOLD, acc, 0.35), fa), tracking=4, stroke=sb)

        # 点位标注（避免重叠）
        if ma > 0.01:
            f = F.get("sansb", 22)
            placed = []

            def put(txt, i, col, above, small=False, side=False):
                fnt = F.get("sans", 18) if small else f
                x, y = self.point(i, p, view)
                if side:
                    draw_text(d, (x + 26, y + 8), txt, fnt, rgba(col, 0.95 * ma), "l", tracking=1)
                    return
                w = text_w(fnt, txt, 1)
                y = y - 32 if above else y + 46
                lx = clamp(x, 160 + w / 2, W - 160 - w / 2)
                for _ in range(4):
                    box = (lx - w / 2 - 8, y - 26, lx + w / 2 + 8, y + 8)
                    if any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in placed):
                        y += -34 if above else 34
                    else:
                        break
                else:
                    return
                placed.append(box)
                draw_text(d, (lx, y), txt, fnt, rgba(col, 0.95 * ma), "m", tracking=1)
            put(f"年内最高 {self.c[hi_i]:,.0f}", hi_i, RED, True)
            put(f"年内最低 {self.c[lo_i]:,.0f}", lo_i, GREEN, False)
            put(f"起点 {self.c[0]:,.0f}", 0, GOLD, False)
            put(f"最新 {self.c[-1]:,.0f}", N - 1, GOLD, True, side=True)
            for (ex, ty, tag, acc) in ev_rows:
                draw_text(d, (ex, ty), tag, F.get("sans", 19), rgba(acc, 0.92 * ma), "m", tracking=2)

        if a.outro:
            lines = a.outro.split("|")
            for j, ln in enumerate(lines):
                draw_text(d, (1824, 930 + 62 * j - 62 * (len(lines) - 1)), ln, F.get("serif", 46),
                          rgba(GOLD, 0.95 * fa), "r", tracking=6, stroke=0 if F.native_bold.get("serif") else 2)
        draw_text(d, (W // 2, 1030), f"数据来源：{a.source} · 前复权日K   |   本视频仅为历史数据回顾，不构成任何投资建议",
                  F.get("sans", 17), rgba(GRAY, 0.6 * foot), "m", tracking=1)
        return self.compose(img, ui)

    # ---------------- 入口 ----------------
    def frame(self, idx, spec):
        sc = spec[0]
        if sc == "title":
            img = self.frame_title(idx, spec)
        elif sc == "main":
            img = self.frame_main(idx, spec)
        else:
            img = self.frame_end(idx, spec)
        ow, oh = self.a.out_w, self.a.out_h
        if (ow, oh) != (W, H):
            img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(img)

    def schedule(self):
        a, N, evs = self.a, self.N, self.evs
        fps = a.fps
        specs = []
        for f in range(int(a.title_sec * fps)):
            specs.append(("title", f))
        mf = 0
        base = a.speed / fps
        p, ei = 0.0, 0
        marks = {"title_end": len(specs) / fps}
        while True:
            if ei < len(evs) and p >= evs[ei]["idx"] - 1e-9:
                marks[f"event{ei + 1}"] = len(specs) / fps
                for hf in range(a.hold_frames):
                    specs.append(("main", float(evs[ei]["idx"]), ei, hf, mf))
                    mf += 1
                ei += 1
                continue
            specs.append(("main", p, -1, 0, mf))
            mf += 1
            if p >= N - 1 - 1e-9:
                break
            target = evs[ei]["idx"] if ei < len(evs) else N - 1
            step = base * (0.22 + 0.78 * smooth((target - p) / 4.0))
            p = min(p + step, target)
        marks["end"] = len(specs) / fps
        for f in range(int(a.end_sec * fps)):
            specs.append(("end", f / fps))
        return specs, marks


# ----------------------------------------------------------------------------
# 数据准备 / 多进程
# ----------------------------------------------------------------------------
NAMES = {"日期": "Date", "开盘": "Open", "收盘": "Close", "最高": "High", "最低": "Low", "成交量": "Volume",
         "date": "Date", "open": "Open", "close": "Close", "high": "High", "low": "Low", "volume": "Volume"}


def prepare_data(a):
    if a.csv == "auto":
        import akshare as ak
        s = (pd.Timestamp(a.start) - pd.Timedelta(days=45)).strftime("%Y%m%d")
        e = pd.Timestamp(a.end or pd.Timestamp.today()).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=a.symbol, period="daily", start_date=s, end_date=e, adjust="qfq")
    else:
        df = pd.read_csv(a.csv)
    df = df.rename(columns={c: NAMES.get(c, NAMES.get(str(c).lower(), c)) for c in df.columns})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    if "Ret" not in df.columns:
        df["Ret"] = df["Close"].pct_change()
    df = df[df["Date"] >= pd.Timestamp(a.start)]
    if a.end:
        df = df[df["Date"] <= pd.Timestamp(a.end)]
    df = df.reset_index(drop=True)
    if len(df) < 10:
        sys.exit(f"数据太少（{len(df)} 行），请检查数据和起止日期")
    if pd.isna(df.loc[0, "Ret"]):
        df.loc[0, "Ret"] = df.loc[0, "Close"] / df.loc[0, "Open"] - 1
    keep = ["Date", "Open", "High", "Low", "Close", "Volume", "Ret"]
    return df[[c for c in keep if c in df.columns]]


_R = None


def _init(a):
    global _R
    _R = Renderer(a)


def _work(job):
    idx, spec = job
    return _R.frame(idx, spec).tobytes()


def main():
    ap = argparse.ArgumentParser(description="贵州茅台日K电影感动画")
    ap.add_argument("csv", nargs="?", default="auto", help="auto=用 akshare 在线取数；或 CSV 路径")
    ap.add_argument("--symbol", default="600519")
    ap.add_argument("--name", default="贵州茅台")
    ap.add_argument("--name-en", dest="name_en", default="KWEICHOW MOUTAI")
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--events", default="events.json")
    ap.add_argument("--out", default="moutai_cinematic_2026.mp4")
    ap.add_argument("--speed", type=float, default=8.0, help="每秒推进多少个交易日")
    ap.add_argument("--no-jitter", dest="jitter", action="store_false", help="关闭折线头部的上下跳动")
    ap.add_argument("--px", type=float, default=16.0, help="滚动模式下每个交易日占多少像素（越大，画面里可见的天数越少）")
    ap.add_argument("--fit", action="store_true", help="改回旧样式：整段走势逐渐压缩，不滚动")
    ap.add_argument("--hold", type=float, default=3.4, help="每个事件信息卡停留秒数")
    ap.add_argument("--title-sec", dest="title_sec", type=float, default=4.5)
    ap.add_argument("--end-sec", dest="end_sec", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--audio", default=None, help="背景音乐文件（mp3/m4a/wav），自动淡入淡出")
    ap.add_argument("--source", default="东方财富")
    ap.add_argument("--outro", default=None, help="片尾金色文案，用 | 分行")
    ap.add_argument("--preview", type=float, default=None, help="只导出第 N 秒的一帧 PNG")
    ap.add_argument("--info", action="store_true", help="只打印时间线")
    ap.add_argument("--font-serif", dest="font_serif", default=None)
    ap.add_argument("--font-sans", dest="font_sans", default=None)
    ap.add_argument("--font-num", dest="font_num", default=None)
    a = ap.parse_args()

    a.out_w, a.out_h = [int(v) for v in a.size.lower().split("x")]
    a.hold_frames = int(a.hold * a.fps)

    df = prepare_data(a)
    a.data_csv = os.path.splitext(a.out)[0] + "_data.csv"
    df.to_csv(a.data_csv, index=False)
    print(f"数据：{len(df)} 个交易日 {df.Date.iloc[0]:%Y-%m-%d} ~ {df.Date.iloc[-1]:%Y-%m-%d}，已保存 {a.data_csv}")

    R = Renderer(a)
    specs, marks = R.schedule()
    total = len(specs) / a.fps
    print(f"总时长约 {total:.1f} 秒，共 {len(specs)} 帧；事件 {len(R.evs)} 个")
    print("时间线(秒)：" + "，".join(f"{k}={v:.1f}" for k, v in marks.items()))
    if a.info:
        return

    if a.preview is not None:
        i = min(int(a.preview * a.fps), len(specs) - 1)
        img = R.frame(i, specs[i])
        png = os.path.splitext(a.out)[0] + f"_preview_{a.preview:g}s.png"
        Image.fromarray(img).save(png)
        print("预览已保存：", png)
        return

    tmp = os.path.splitext(a.out)[0] + ".part.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{a.out_w}x{a.out_h}", "-r", str(a.fps), "-i", "-"]
    if a.audio:
        cmd += ["-i", a.audio, "-map", "0:v", "-map", "1:a", "-af",
                f"afade=t=in:d=1.5,afade=t=out:st={max(total - 3.5, 0):.2f}:d=3.5", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except FileNotFoundError:
        sys.exit("找不到 ffmpeg，请先安装并加入 PATH")

    # 每个渲染进程约占 300~500MB 内存，默认最多 3 个；内存紧张时用 --workers 1
    workers = a.workers or max(1, min((os.cpu_count() or 2) // 2, 3))
    print(f"渲染进程数：{workers}")
    jobs = list(enumerate(specs))
    try:
        if workers > 1:
            import multiprocessing as mp
            batch = workers * 3   # 分批取回，避免已渲染的帧在内存里越积越多
            with mp.Pool(workers, initializer=_init, initargs=(a,)) as pool:
                for s0 in range(0, len(jobs), batch):
                    for buf in pool.map(_work, jobs[s0:s0 + batch], chunksize=1):
                        proc.stdin.write(buf)
                    if (s0 // batch) % 10 == 0:
                        print(f"{s0}/{len(jobs)}", flush=True)
        else:
            for n, (i, s) in enumerate(jobs):
                proc.stdin.write(R.frame(i, s).tobytes())
                if n % 60 == 0:
                    print(f"{n}/{len(jobs)}", flush=True)
    except (BrokenPipeError, OSError):
        proc.wait()
        sys.exit("\nffmpeg 中途退出（常见原因：内存不足）。请关闭其他程序后，改用 "
                 "--workers 1 --size 1280x720 重试；上面 ffmpeg 的报错信息也请发给我。")
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
    if proc.wait() == 0:
        os.replace(tmp, a.out)
        print("完成：", a.out)
    else:
        sys.exit("ffmpeg 失败")


if __name__ == "__main__":
    main()