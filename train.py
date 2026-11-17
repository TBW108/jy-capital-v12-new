# -*- coding: utf-8 -*-
"""v12_new: raw-minute end-to-end cross-sectional model and original training recipe.

Shared preprocessing, architecture, training and checkpoint definitions live here.
The archived numerical implementation is preserved. Runtime data access/logging and
command-line entry points are adapted for this standalone repository. See README.md
for the input schema, competition rules, architecture, and reproduction instructions.
Training dates are fixed independently of inference dates. No market data is bundled.
"""
import json
import os
import time
import warnings

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from runtime import dai, get_logger

logger = get_logger()

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "model.json")

# ---------- 配置(写死,不随平台入参变化) ----------
TRAIN_START, TRAIN_END = "2020-01-01", "2024-12-31 23:59:59"  # 训练区间写死,切勿用平台注入的测试区间训练
LABEL_TABLE = "cpt_jyc_2026_vwap"
SECTION_TABLE = "cpt_jyc_2026_instruments"
# bar1m 表里除复权因子外的全部 25 个原始字段(v5 及以前只用了每组的前几个,即官方 demo 的 10 个字段)
PRICE_COLS = ["open", "high", "low", "close", "bid_price1", "ask_price1",
              "bid_price2", "ask_price2", "bid_price3", "ask_price3"]
VOL_COLS = ["volume", "amount", "bid_volume1", "ask_volume1",
            "bid_volume2", "ask_volume2", "bid_volume3", "ask_volume3"]
CNT_COLS = ["deal_number", "bid_num_orders1", "ask_num_orders1",
            "bid_num_orders2", "ask_num_orders2", "bid_num_orders3", "ask_num_orders3"]   # 笔数类字段
BAR_COLS = PRICE_COLS + VOL_COLS + CNT_COLS  # 模型输入的全部通道,顺序固定
N_CH = len(BAR_COLS)

N_AUC = 10                    # 竞价分钟位:0..9 = 09:16 … 09:25
N_POS = N_AUC + 240           # 一天的分钟位总数;10..249 = 09:31 … 15:00
PATCH = 5                     # 每个 patch 覆盖的分钟数(线性 embedding 的输入窗口)
SEG = 30                      # 每个 token 覆盖的分钟数:一个半小时段内的 patch embedding 池化成一个 token
HIST_DAYS = 20                # 额外喂给模型的历史交易日数(同样是原始 1 分钟 bar);v8:1 -> 20
HIST_FINE = 1                 # 其中最近 HIST_FINE 天每天保留 8 个半小时 token,更早的每天压成 1 个 token
VOL_WIN = 20                  # 量的标准化尺度:此前 VOL_WIN 个交易日的分钟均量
PRICE_CLIP, VOL_CLIP = 30.0, 10.0           # 标准化后的截断范围(价格单位是 %)
BUFFER_DAYS = 45              # 推理/训练区间之前多取的日历天数,给标准化尺度和历史窗口用
CHUNK_DAYS = 10               # 每次向 dai 取多少个日历天的 1 分钟 bar
MIN_NAMES = 30                # 一个截面至少要有这么多只股票才参与损失
MAX_ABS_LABEL = 0.25          # |vwap_return| 超过它视为坏数据
N_JOBS = int(os.environ.get("JY_N_JOBS", "1"))   # 本地可并行取数;平台上保持 1

SEEDS = (42, 43, 44, 45, 46)  # 多个种子取平均,降低预测方差;v11:3 -> 5 个(加了分组模块后,不同种子之间的差别变大)
STEPS, BATCH_DAYS, LR, WD = 4000, 8, 1e-3, 1e-2    # v6_tune:学习率 2e-3 -> 1e-3(与 2 层编码器搭配,验证集上最好)
IR_PENALTY = 0.5              # 损失 = −mean(IC) + IR_PENALTY·std(IC)
TURN_PENALTY = 0.06           # 损失里奖励相邻截面分数的截面相关(降换手);与 v5_tune 相同
SMOOTH = 0.0                  # 推理时截面间指数平滑系数;0 = 不平滑
AUX_WEIGHT = 0.5              # v12:辅助目标(下一窗口的异常成交量)在损失里的权重;只用于训练,推理不用

MODEL_CFG = dict(n_ch=N_CH, patch=PATCH, seg=SEG, n_auc=N_AUC, d_model=64, nhead=4, nlayers=2, dim_ff=128,
                 dropout=0.1, hist_days=HIST_DAYS, hist_fine=HIST_FINE, peer=True, peer_ff=1,
                 run=False, xmean=True, moe_k=8, aux=1)
# v12 = v11 + 两处改动:
#   1) 复权:用原始字段 adjust_factor 把昨收、历史价格和历史成交量换算到当天的尺度(除权除息日不再出现假跳空);
#      adjust_factor 只用于对齐,不作为模型输入
#   2) 辅助目标(aux=1):训练时多预测"下一个窗口的异常成交量"(官方标签表的 volume 相对此前 20 日均量),
#      只进入训练损失,推理时不用
# v10 = v9 把类型数从 4 改成 8(三组随机种子的四折验证里 IC_IR 平均更高,见 ARCHITECTURE.md 9.8 节)。
# v9 相对 v8:在第二次股票间注意力之后、输出之前加一个"按类型分组"的模块(moe_k = 类型数,0 = 不加)。
#   1) 每只股票按它过去 19 天的摘要(较早交易日的 token 经过时间编码器后的平均,再做截面标准化)得到一组"属于哪一类"的权重;
#   2) 把"同一类股票的平均编码"加回给每只股票(类内的共同走势);
#   3) moe_k 个专家各自处理,按权重合成。
#   分类权重和专家都由损失自己学出来,不使用行业或风险因子数据;只依赖当前及更早的 token,不破坏因果性。
# v8 相对 v6_tune 的结构变化(依据见 TECH_REPORT_v6_tune.md 7.6 节和 ARCHITECTURE.md 第 9 节):
#   run=False    去掉"到目前为止的全天概况"(累计均值/最大/最小):时间编码器自己就能回看当天之前的 token;
#                两组随机种子的四折验证里,去掉后 IC_IR 都高 2~3%
#   xmean=True   保留两处"和全市场比"(MeanLayer):连它一起去掉时 IC_IR 不再提高
#   hist_days    1 -> 20:昨天保留 8 个半小时 token,更早的 19 天每天压成 1 个 token("那天大致怎么样")
DATA_ALIGNMENT = "section_reads_only_tokens_whose_bars_end_le_section_time; scales_use_previous_days_only"


# ---------- 运行环境 ----------
def limit_threads(cap=8):
    """把 torch 的 CPU 线程数限制在 cap 以内。容器里 torch 默认按宿主机核数开线程,会严重抢占。"""
    n = os.cpu_count() or 1
    try:
        n = min(n, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    n = max(1, min(n, cap))
    try:
        torch.set_num_threads(n)
    except RuntimeError:
        pass
    return n

# ---------- 取数 ----------
def _ts(x):
    return pd.Timestamp(x).strftime("%Y-%m-%d %H:%M:%S")


def day_end(x):
    """区间右端点:只给了日期(时间恰为 00:00:00)时按当天收盘后处理,否则会漏掉最后一个交易日。"""
    t = pd.Timestamp(x)
    return t + pd.Timedelta(hours=23, minutes=59, seconds=59) if t == t.normalize() else t


def load_sections(sd, ed):
    """比赛的样本主键:每个交易日的若干日内截面时点 × 股票池。"""
    df = dai.query(f"SELECT date, instrument FROM {SECTION_TABLE}",
                   filters={"date": [_ts(sd), _ts(day_end(ed))]}).df()
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna().drop_duplicates(["date", "instrument"]).reset_index(drop=True)


def load_labels(sd, ed):
    try:                                                        # volume 只用于辅助目标;平台表里没有这一列时退回只取标签
        df = dai.query(f"SELECT date, instrument, vwap_return, volume FROM {LABEL_TABLE}",
                       filters={"date": [_ts(sd), _ts(ed)]}).df()
    except Exception:
        df = dai.query(f"SELECT date, instrument, vwap_return FROM {LABEL_TABLE}",
                       filters={"date": [_ts(sd), _ts(ed)]}).df()
    df["date"] = pd.to_datetime(df["date"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["vwap_return"])
    return df.drop_duplicates(["date", "instrument"]).reset_index(drop=True)


def load_bars(table, sd, ed, instruments):
    sql = f"SELECT date, instrument, {', '.join(BAR_COLS + ['adjust_factor'])} FROM {table}"   # 复权因子只用于对齐尺度
    df = dai.query(sql, filters={"date": [_ts(sd), _ts(ed)], "instrument": list(instruments)}).df()
    df["date"] = pd.to_datetime(df["date"])
    return df


def minute_index(ts):
    """时间戳 -> 日内分钟格。bar 以结束时刻标记:09:31→1 … 11:30→120,13:01→121 … 15:00→240;
    09:30 及之前(集合竞价)→0。截面时点用同一映射,所以"bar 格 <= 截面格"即不含未来。"""
    ts = pd.DatetimeIndex(ts)
    m = (ts.hour * 60 + ts.minute).to_numpy()
    return np.where(m <= 570, 0, np.where(m <= 690, m - 570, np.where(
        m <= 780, 120, np.where(m <= 900, m - 660, 240)))).astype(np.int64)


def _ffill(a):
    """沿最后一维前向填充 NaN(开头的 NaN 保持)。"""
    idx = np.where(np.isfinite(a), np.arange(a.shape[-1]), 0)
    np.maximum.accumulate(idx, axis=-1, out=idx)
    return np.take_along_axis(a, idx, axis=-1)

# ---------- 标签处理与通用工具 ----------
def gauss_rank(Y, M):
    """逐截面把标签换成 秩 -> 标准正态分位;无效处为 0。Y:(…, I)。"""
    ok = M & np.isfinite(Y)
    order = np.argsort(np.where(ok, Y, np.inf), axis=-1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(Y.shape[-1]), Y.shape).copy(), -1)
    n = ok.sum(-1, keepdims=True).clip(min=1)
    p = np.clip((rank + 0.5) / n, 1e-4, 1 - 1e-4)
    z = (2.0 ** 0.5) * torch.erfinv(torch.from_numpy(2 * p - 1)).numpy()
    return np.where(ok, z, 0.0).astype(np.float32), ok


def _roll(x, w):
    """沿第 0 维(交易日)的滚动 和/计数,窗口是**严格之前**的 w 天,NaN 忽略。"""
    f = np.isfinite(x)
    cs = np.concatenate([np.zeros_like(x[:1], np.float64), np.cumsum(np.where(f, x, 0.0), 0, dtype=np.float64)])
    cn = np.concatenate([np.zeros_like(x[:1], np.float64), np.cumsum(f, 0, dtype=np.float64)])
    i = np.arange(x.shape[0])
    lo = np.clip(i - w, 0, None)
    return cs[i] - cs[lo], cn[i] - cn[lo]


def _rmean(x, w, min_n):
    s, n = _roll(x, w)
    return np.where(n >= min_n, s / np.maximum(n, 1), np.nan)


def _shift(x):
    return np.concatenate([np.full_like(x[:1], np.nan), x[:-1]])


def xs_corr(p, y, m):
    """逐截面 Pearson 相关。p,y,m:(…, I) -> (…)"""
    n = m.sum(-1, keepdim=True).clamp(min=1)
    z = torch.zeros((), dtype=p.dtype, device=p.device)
    pc = torch.where(m, p - torch.where(m, p, z).sum(-1, keepdim=True) / n, z)
    yc = torch.where(m, y - torch.where(m, y, z).sum(-1, keepdim=True) / n, z)
    return (pc * yc).sum(-1) / (pc.norm(dim=-1) * yc.norm(dim=-1) + 1e-8)


# ---------- 预处理:一个时间块的 1 分钟 bar -> 逐 (日, 股票) 的原始分钟张量 ----------
def _chunk_raw(df):
    """返回 days, insts, G (nd,ni,N_POS,N_CH) float16, daily (nd,ni,5)。

    G 的价格通道存 价格/当日第一个价格 − 1,量和笔数通道存 log1p(原始值);真正的标准化(除以昨收、
    除以历史均量 / 历史均笔数)要用到前几天的数据,留到 build_panel / make_inputs 里做。
    daily = [当日第一个价格 a0, 当日最后价格, 连续竞价总成交量, 第一个有价格的分钟位, 连续竞价总成交笔数]。"""
    if len(df) == 0:
        return None
    ts = df["date"].to_numpy("datetime64[ns]")
    order = np.argsort(ts, kind="stable")
    ts = ts[order]
    days, di = np.unique(ts.astype("datetime64[D]"), return_inverse=True)
    codes, insts = pd.factorize(df["instrument"].to_numpy()[order], sort=True)
    t_idx = pd.DatetimeIndex(ts)
    hm = (t_idx.hour * 60 + t_idx.minute).to_numpy()
    pos = np.where(hm <= 570, np.clip(hm - 556, 0, N_AUC - 1), N_AUC - 1 + minute_index(ts))
    nd, ni = len(days), len(insts)
    lin = (di * ni + codes) * N_POS + pos
    size = nd * ni * N_POS

    def col(c):
        return np.nan_to_num(df[c].to_numpy(np.float64)[order], nan=0.0, posinf=0.0, neginf=0.0)

    def last(c, positive):                                   # 同一分钟位有多根 bar 时取最后一根
        a = np.full(size, np.nan if positive else 0.0)
        a[lin] = col(c) if positive else np.maximum(col(c), 0)
        if positive:
            a[~(a > 0)] = np.nan
        return a.reshape(nd, ni, N_POS)

    def total(c):
        return np.bincount(lin, weights=np.maximum(col(c), 0), minlength=size).reshape(nd, ni, N_POS)

    prc = {c: last(c, True) for c in PRICE_COLS}
    high, low, close = prc["high"], prc["low"], prc["close"]
    vol, amt, deals = total("volume"), total("amount"), total("deal_number")   # 分钟内发生的量:同一分钟位累加
    book = [last(c, False) for c in VOL_COLS[2:]]            # 盘口挂单量:分钟末快照
    cnt = [last(c, False) for c in CNT_COLS[1:]]             # 盘口委托笔数:分钟末快照
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # 成交额的计量单位(只取 10 的整数次幂)与坏点(隐含均价明显越界的 bar 用 量×收盘价 重建)
        ok = (vol > 0) & (close > 0)
        k = np.nanmedian((amt / (vol * close))[ok]) if ok.any() else 1.0
        if np.isfinite(k) and k > 0:
            amt = amt / 10.0 ** np.round(np.log10(k))
        px = amt / vol
        bad = ok & ((px < 0.9 * low) | (px > 1.1 * high))
        amt[bad] = (vol * close)[bad]

        C = _ffill(close)
        has = np.isfinite(C)
        fp = np.where(has.any(-1), has.argmax(-1), N_POS)    # 第一个有价格的分钟位
        a0 = np.take_along_axis(C, np.minimum(fp, N_POS - 1)[..., None], -1)[..., 0]
        C = np.where(has, C, a0[..., None])
        fill = lambda x: np.where(np.isfinite(x), x, C)      # 缺失的价格沿用最近的收盘价
        Pr = np.stack([C if c == "close" else fill(prc[c]) for c in PRICE_COLS], -1) / a0[..., None, None] - 1
        G = np.concatenate([np.clip(Pr, -0.9, 9.0), np.log1p(np.stack([vol, amt] + book + [deals] + cnt, -1))], -1)
    G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        af = np.nanmax(last("adjust_factor", True), -1)       # 当日的复权因子(日内不变)
    daily = np.stack([a0, C[:, :, -1], vol[:, :, N_AUC:].sum(-1), fp.astype(np.float64),
                      deals[:, :, N_AUC:].sum(-1), af], -1)
    return {"days": days, "insts": np.asarray(insts, dtype=object), "G": G, "daily": daily}


def _chunk_job(args):
    table, a, b, insts = args
    return _chunk_raw(load_bars(table, a, b, insts))


# ---------- 面板:取数 + 对齐 + 标准化尺度 ----------
def build_panel(table, sd, ed, with_labels):
    """返回 [sd − BUFFER_DAYS, ed] 上的稠密面板(缓冲段只给标准化尺度和历史窗口用)。

        G     (D,I,N_POS,N_CH) float16 原始分钟张量(见 _chunk_raw)
        pr    (D,I) 当日第一个价格 / 昨收        ref  (D,I) 昨收(没有则为当日第一个价格)
        vs    (D,I) 此前 VOL_WIN 日的分钟均量     a0   (D,I) 当日第一个价格
        ns    (D,I) 此前 VOL_WIN 日的分钟均成交笔数
        fp    (D,I) 当日第一个有价格的分钟位         insec (D,I) 当天是否在截面表里
        M     (D,K,I) 该 (日,截面,股票) 在截面表里且截面时点已有价格
        Y     (D,K,I) 官方标签(with_labels 才有)
        days (D,)  grid (K,) 截面分钟格  read (K,) 每个截面读取的 token 下标  insts (I,)  in_range (D,)
    """
    t0 = time.time()
    sd, ed = pd.Timestamp(sd), day_end(ed)
    buf = sd.normalize() - pd.Timedelta(days=BUFFER_DAYS)
    sec = load_sections(buf, ed)
    if len(sec) == 0:
        raise RuntimeError(f"截面表在 {buf} ~ {ed} 内没有数据")
    sec_day = sec["date"].to_numpy("datetime64[D]")
    sec_mi = minute_index(sec["date"])
    grid = np.unique(sec_mi)
    insts = np.array(sorted(sec["instrument"].unique()), dtype=object)

    edges = pd.date_range(buf, ed.normalize(), freq=f"{CHUNK_DAYS}D")
    jobs = []
    for a in edges:
        b = min(a + pd.Timedelta(days=CHUNK_DAYS) - pd.Timedelta(seconds=1),
                ed.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59))
        in_chunk = (sec_day >= a.to_datetime64().astype("datetime64[D]")) & \
                   (sec_day <= b.to_datetime64().astype("datetime64[D]"))
        sub = np.unique(sec["instrument"].to_numpy()[in_chunk]) if in_chunk.any() else insts
        jobs.append((table, a, b, list(sub)))
    if N_JOBS > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(N_JOBS) as pool:
            chunks = pool.map(_chunk_job, jobs)
    else:
        chunks = [_chunk_job(j) for j in jobs]
    chunks = [c for c in chunks if c is not None]
    if not chunks:
        raise RuntimeError(f"{table} 在 {buf} ~ {ed} 内没有 1 分钟 bar")

    days = np.unique(np.concatenate([c["days"] for c in chunks] + [np.unique(sec_day)]))
    nD, K, nI = len(days), len(grid), len(insts)
    G = np.zeros((nD, nI, N_POS, N_CH), np.float16)
    daily = np.full((nD, nI, 6), np.nan)
    for c in chunks:
        d_idx = np.searchsorted(days, c["days"])
        pos = np.searchsorted(insts, c["insts"])
        keep = (pos < nI) & (insts[np.clip(pos, 0, nI - 1)] == c["insts"])
        ix = np.ix_(d_idx, pos[keep])
        G[ix] = c["G"][:, keep]
        daily[ix] = c["daily"][:, keep]
    del chunks

    # ---- 标准化尺度:只用到前一日及更早的数据
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a0, last, vday, fp, nday, af = [daily[:, :, j] for j in range(6)]
        # 复权:除权除息日的昨收换算到当天的价格尺度(分红让价格变低、送转让价格减半),否则会看到假的跳空
        af = _ffill(np.where(af > 0, af, np.nan).T).T
        af = np.where(np.isfinite(af) & (af > 0), af, 1.0)
        ratio = _shift(af) / af
        ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, 1.0)
        ref = _shift(last) * ratio
        ref = np.where(ref > 0, ref, a0)                          # 没有昨收:用当日第一个价格

        def minute_scale(day_total):                              # 此前 VOL_WIN 日的分钟均值(_rmean 不含当日)
            sc = _rmean(np.where(day_total > 0, day_total, np.nan), VOL_WIN, 3) / 240.0
            med = np.nanmedian(sc, 1, keepdims=True)              # 没有历史的股票:用当日全市场中位尺度
            med = np.where(np.isfinite(med), med, np.nanmedian(np.where(day_total > 0, day_total, np.nan)) / 240.0)
            return np.where(sc > 0, sc, np.where(np.isfinite(med), med, 1.0))

        vs, ns = minute_scale(vday / af) * af, minute_scale(nday)   # 成交量按股本变化换算到当天的尺度
        pr = a0 / ref

    # ---- 截面掩码 / 标签
    def to_grid(day, mi, ins):
        d, k, i = np.searchsorted(days, day), np.searchsorted(grid, mi), np.searchsorted(insts, ins)
        ok = (i < nI) & (insts[np.clip(i, 0, nI - 1)] == ins)
        return d[ok], k[ok], i[ok], ok

    M = np.zeros((nD, K, nI), bool)
    d, k, i, _ = to_grid(sec_day, sec_mi, sec["instrument"].to_numpy())
    M[d, k, i] = True
    insec = M.any(1)                                             # 当天在截面表里的股票(事前已知)
    M &= np.nan_to_num(fp, nan=N_POS)[:, None, :] <= (N_AUC - 1 + grid)[None, :, None]   # 截面时点必须已有价格
    Y = V = None
    if with_labels:
        lab = load_labels(sd, ed)
        Y = np.full((nD, K, nI), np.nan, np.float32)
        d, k, i, ok = to_grid(lab["date"].to_numpy("datetime64[D]"), minute_index(lab["date"]),
                              lab["instrument"].to_numpy())
        Y[d, k, i] = lab["vwap_return"].to_numpy(np.float32)[ok]
        Y[np.abs(Y) > MAX_ABS_LABEL] = np.nan                    # 标签表里的坏点
        if "volume" in lab.columns:                              # 下一窗口成交量:只用于辅助目标
            V = np.full((nD, K, nI), np.nan, np.float32)
            V[d, k, i] = lab["volume"].to_numpy(np.float32)[ok]
    read = grid // SEG                                           # 截面 s 读第 s//SEG 个 token(只含分钟位 < N_AUC + s 的 bar)
    in_range = days >= sd.normalize().to_datetime64().astype("datetime64[D]")
    logger.info("面板构建完成", days=int(nD), in_range=int(in_range.sum()), sections=int(K),
                instruments=int(nI), cells=int(M.sum()), elapsed=round(time.time() - t0, 1))
    f32 = lambda x: np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return {"G": G, "pr": f32(pr), "ref": f32(ref), "vs": f32(vs), "ns": f32(ns), "a0": f32(a0), "valid": np.isfinite(a0) & (a0 > 0),
            "fp": np.nan_to_num(fp, nan=N_POS).astype(np.int16), "insec": insec, "M": M, "Y": Y, "V": V, "af": f32(af), "days": days, "grid": grid, "read": read, "insts": insts, "in_range": in_range}


TENSOR_KEYS = ("G", "pr", "ref", "vs", "ns", "a0", "af", "valid", "fp", "insec", "M")


def to_tensors(P, device=None, keep_cpu=()):
    """面板里模型要用的数组 -> torch 张量(可选搬到 device;keep_cpu 里的键留在内存)。"""
    out = dict(P)
    for k in TENSOR_KEYS:
        out[k] = torch.as_tensor(P[k]) if device is None or k in keep_cpu else torch.as_tensor(P[k]).to(device)
    return out


def _normalize(g, p_scale, v_scale, a_scale, n_scale, first):
    """原始分钟张量 -> 网络输入。g:(N,T,N_CH);四个尺度都是 (N,);first:(N,) 第一个有价格的分钟位。

    价格通道:价格/昨收 − 1,单位 %;量通道:log1p(量 / 历史分钟均量),成交额另除以昨收;
    笔数通道:log1p(笔数 / 历史分钟均成交笔数)。
    还没有价格的分钟位(first 之前)价格通道记 0,即"沿用昨收" —— 不能让后面才出现的价格提前露出来。"""
    np_ = len(PRICE_COLS)
    price = ((1.0 + g[..., :np_]) * p_scale[:, None, None] - 1.0) * 100.0
    seen = torch.arange(g.shape[1], device=g.device)[None, :] >= first[:, None]
    price = torch.where(seen[..., None], price, torch.zeros_like(price))
    raw = torch.expm1(g[..., np_:].clamp(max=40.0))
    den = torch.stack([v_scale, a_scale] + [v_scale] * (len(VOL_COLS) - 2) + [n_scale] * len(CNT_COLS),
                      -1)[:, None, :].clamp(min=1e-12)
    return torch.cat([price.clamp(-PRICE_CLIP, PRICE_CLIP), torch.log1p(raw / den).clamp(0.0, VOL_CLIP)], -1)


def make_inputs(P, d, device, hist_days=0):
    """P:to_tensors 的结果;d:(B,) 天下标。
    返回 x (N,N_POS,N_CH)、hist (N,H,240,N_CH)|None、hmask (N,H)|None(该历史日是否有行情)、b_idx、i_idx。"""
    b_idx, i_idx = torch.nonzero(P["insec"][d], as_tuple=True)   # 当天截面表里的全部股票(与之后有没有行情无关)
    dd = d[b_idx]
    ref, vs, ns = P["ref"][dd, i_idx].to(device), P["vs"][dd, i_idx].to(device), P["ns"][dd, i_idx].to(device)
    G, gd = P["G"], P["G"].device                               # 原始面板可以留在内存里,其余张量在显卡上
    x = _normalize(G[dd.to(gd), i_idx.to(gd)].to(device).float(), P["pr"][dd, i_idx].to(device), vs, vs * ref, ns,
                   P["fp"][dd, i_idx].to(device).long())
    hist = hmask = None
    if hist_days > 0:
        parts, oks = [], []
        for j in range(1, hist_days + 1):
            dj = dd - j
            okj = (dj >= 0)
            djc = dj.clamp(min=0)
            okj = (okj & P["valid"][djc, i_idx]).to(device)
            g = G[djc.to(gd), i_idx.to(gd)][:, N_AUC:].to(device).float()
            a0j = P["a0"][djc, i_idx].to(device) * P["af"][djc, i_idx].to(device) / P["af"][dd, i_idx].to(device)   # 换算到当天的价格尺度
            h = _normalize(g, a0j / ref.clamp(min=1e-12), vs, vs * ref, ns,
                           (P["fp"][djc, i_idx].to(device).long() - N_AUC).clamp(min=0))
            parts.append(torch.where(okj[:, None, None], h, torch.zeros_like(h)))
            oks.append(okj)
        hist, hmask = torch.stack(parts[::-1], 1), torch.stack(oks[::-1], 1)   # 时间从早到晚
    return x, hist, hmask, b_idx, i_idx


# ---------- 模型:原始分钟 bar 上的因果 Transformer ----------
def group_mean(h, gid, n_groups):
    """按组(同一天的全部股票)求均值。h:(N,…) gid:(N,) -> (n_groups,…)"""
    s = torch.zeros((n_groups,) + h.shape[1:], dtype=h.dtype, device=h.device).index_add_(0, gid, h)
    n = torch.zeros(n_groups, dtype=h.dtype, device=h.device).index_add_(0, gid, torch.ones_like(gid, dtype=h.dtype))
    return s / n.clamp(min=1).reshape((-1,) + (1,) * (h.dim() - 1))


def group_std(h, m, gid, n_groups):
    """按组求标准差。m 是 group_mean(h)[gid]。"""
    return group_mean((h - m) ** 2, gid, n_groups).clamp(min=1e-6).sqrt()


class MeanLayer(nn.Module):
    """截面均值层:每只股票拿到"自己相对全市场的偏离(除以截面标准差)""全市场均值"和"截面离散度"。
    当天股票之间分化有多大,由网络自己从这三个量里感知,不需要人为区分行情。"""

    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(3 * d, d)

    def forward(self, h, gid, n_groups):
        m = group_mean(h, gid, n_groups)[gid]
        s = group_std(h, m, gid, n_groups)[gid]
        return h + self.lin(torch.cat([(h - m) / s, m, s], -1))


class PeerBlock(nn.Module):
    """同一个截面内,股票之间的注意力:每只股票对当天截面里的全部股票做一次多头注意力。

    层里没有任何股票身份(对股票的排列等变):一只股票参照谁、参照多少,完全由各自到当前为止的
    原始走势编码决定。逐 token 位置计算,所以不破坏因果性。"""

    def __init__(self, d, nhead, dropout, ff=1):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Linear(ff * d, d))

    def forward(self, h, gid, n_groups):
        N, L, d = h.shape                                        # gid 必须已按组排好序(nonzero 的顺序即是)
        cnt = torch.bincount(gid, minlength=n_groups)
        pos = torch.arange(N, device=h.device) - (torch.cumsum(cnt, 0) - cnt)[gid]
        nmax = int(cnt.max())
        x = h.new_zeros(n_groups, nmax, L, d)
        x[gid, pos] = h
        pad = torch.ones(n_groups, nmax, dtype=torch.bool, device=h.device)
        pad[gid, pos] = False
        x = x.permute(0, 2, 1, 3).reshape(n_groups * L, nmax, d)                 # 每个 (交易日, token) 是一个集合
        kpm = pad[:, None, :].expand(-1, L, -1).reshape(n_groups * L, nmax)
        y = self.n1(x)
        x = x + self.attn(y, y, y, key_padding_mask=kpm, need_weights=False)[0]
        x = x + self.ff(self.n2(x))
        return x.reshape(n_groups, L, nmax, d).permute(0, 2, 1, 3)[gid, pos]


class RawFormer(nn.Module):
    """输入一只股票一天的原始分钟 bar(N_POS × N_CH),输出当天每个半小时 token 上的分数。

    每 patch 根 bar 先过同一个 embedding;再按时间段做 均值/最大/最小/末值 池化成 token:
    token 0 = 集合竞价段,token k (k>=1) = 第 k 个半小时。注意力是因果的,所以第 k 个输出只依赖
    第 0..k 个 token(以及更早交易日的历史 token)。同一天内的股票通过"截面均值层"和"股票之间的
    注意力层"互相可见,两者都是逐 token 位置做的,不破坏因果性。"""

    def __init__(self, n_ch=N_CH, patch=PATCH, seg=SEG, n_auc=N_AUC, d_model=64, nhead=4, nlayers=2, dim_ff=128,
                 dropout=0.1, hist_days=0, hist_fine=0, peer=True, peer_ff=1, run=False, xmean=False, moe_k=0, aux=0):
        super().__init__()
        self.aux_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1)) if aux else None   # 只用于训练
        self.last_aux = None
        self.moe_k = moe_k
        if moe_k:
            self.moe_n = nn.LayerNorm(d_model)
            self.moe_gate = nn.Linear(d_model, moe_k)                       # 过去若干天的摘要 -> 属于各类的权重
            self.moe_ctx = nn.Linear(d_model, d_model)                      # 同类股票的平均编码
            self.moe_w1 = nn.Parameter(torch.randn(moe_k, d_model, d_model) * d_model ** -0.5)
            self.moe_b1 = nn.Parameter(torch.zeros(moe_k, d_model))
            self.moe_w2 = nn.Parameter(torch.zeros(moe_k, d_model, d_model))   # 初始为 0:训练起点与不加这个模块时相同
            self.moe_b2 = nn.Parameter(torch.zeros(moe_k, d_model))
            nn.init.zeros_(self.moe_ctx.weight)
            nn.init.zeros_(self.moe_ctx.bias)
        self.patch, self.seg, self.n_auc = patch, seg, n_auc
        self.hist_days, self.hist_fine = hist_days, min(hist_fine, hist_days)
        self.n_tok = 1 + 240 // seg
        self.embed = nn.Sequential(nn.Linear(patch * n_ch, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.segpool = nn.Linear(4 * d_model, d_model)          # 段内 均值/最大/最小/末值 池化(当日与历史共用)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tok, d_model))
        self.pool = nn.Linear(4 * d_model, d_model) if run else None       # 因果的累计 均值/最大/最小 池化
        self.cross = MeanLayer(d_model) if xmean else None
        self.peer0 = PeerBlock(d_model, nhead, dropout, peer_ff) if peer else None      # 时间编码之前的股票间注意力
        # 历史交易日:最近 hist_fine 天每天 240/seg 个 token,更早的每天 1 个 token
        self.n_hist = self.hist_fine * (240 // seg) + (hist_days - self.hist_fine)
        self.hpos = nn.Parameter(torch.zeros(1, max(self.n_hist, 1), d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, nlayers, enable_nested_tensor=False)
        self.cross2 = MeanLayer(d_model) if xmean else None
        self.peer1 = PeerBlock(d_model, nhead, dropout, peer_ff) if peer else None      # 时间编码之后的股票间注意力
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        nn.init.normal_(self.pos, std=0.02)

    def pooled(self, e):
        """e:(N,S,P,d) 每段 P 个 patch embedding -> (N,S,d)"""
        return self.segpool(torch.cat([e.mean(2), e.amax(2), e.amin(2), e[:, :, -1]], -1))

    def forward(self, x, gid, n_groups, hist, hmask):
        """x:(N,T,C) 当日前 T 个分钟位  gid:(N,) 所属交易日在 batch 内的编号
        hist:(N,H,240,C) 与 hmask:(N,H)(hist_days=0 时不使用) -> (N,L),L = 1 + 完整的半小时段数"""
        N, d = x.shape[0], self.pos.shape[-1]
        na, sp = self.n_auc // self.patch, self.seg // self.patch
        e = self.embed(x.reshape(N, x.shape[1] // self.patch, -1))               # (N,patch 数,d)
        n_seg = (e.shape[1] - na) // sp
        toks = [self.pooled(e[:, None, :na])]
        if n_seg > 0:
            toks.append(self.pooled(e[:, na:na + n_seg * sp].reshape(N, n_seg, sp, d)))
        h = torch.cat(toks, 1)
        L = h.shape[1]
        h = h + self.pos[:, :L]
        if self.pool is not None:
            mx, mn = [h[:, 0]], [h[:, 0]]                        # 逐 token 递推的累计最大/最小(因果)
            for j in range(1, L):
                mx.append(torch.maximum(mx[-1], h[:, j]))
                mn.append(torch.minimum(mn[-1], h[:, j]))
            steps = torch.arange(1, L + 1, device=x.device, dtype=h.dtype)[None, :, None]
            h = h + self.pool(torch.cat([h, h.cumsum(1) / steps, torch.stack(mx, 1), torch.stack(mn, 1)], -1))
        if self.cross is not None:
            h = self.cross(h, gid, n_groups)
        if self.peer0 is not None:
            h = self.peer0(h, gid, n_groups)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)
        if self.n_hist > 0:
            H, fine = hist.shape[1], self.hist_fine
            eh = self.embed(hist.reshape(N, H, 240 // self.patch, -1))            # (N,H,48,d)
            eh = eh * hmask[:, :, None, None].to(eh.dtype)                        # 没有行情的历史日记 0
            parts = []
            if H > fine:
                parts.append(self.pooled(eh[:, :H - fine]))                       # 较早的每天 1 个 token
            if fine > 0:
                parts.append(self.pooled(eh[:, H - fine:].reshape(N, fine * (240 // self.seg), sp, d)))
            h = torch.cat([torch.cat(parts, 1) + self.hpos, h], 1)
            full = torch.zeros(self.n_hist + L, self.n_hist + L, dtype=torch.bool, device=x.device)
            full[:self.n_hist, self.n_hist:] = True              # 历史 token 看不到当日
            full[self.n_hist:, self.n_hist:] = causal
            causal = full
        hh = self.encoder(h, mask=causal)
        h = hh[:, -L:]
        if self.cross2 is not None:
            h = self.cross2(h, gid, n_groups)
        if self.peer1 is not None:
            h = self.peer1(h, gid, n_groups)
        if self.moe_k:
            n_old = self.n_hist - self.hist_fine * (240 // self.seg)            # 较早交易日的 token 个数
            s = hh[:, :n_old].mean(1) if n_old > 0 else hh[:, :max(self.n_hist, 1)].mean(1)
            m = group_mean(s, gid, n_groups)[gid]
            s = (s - m) / group_std(s, m, gid, n_groups)[gid]                   # 截面标准化:去掉全市场共有的部分
            w = torch.softmax(self.moe_gate(s).float(), -1).to(h.dtype)         # (N,K):一只股票一天一组权重
            K = self.moe_k
            num = torch.zeros(n_groups, K, L, d, dtype=h.dtype, device=h.device).index_add_(0, gid, w[:, :, None, None] * h[:, None])
            den = torch.zeros(n_groups, K, dtype=h.dtype, device=h.device).index_add_(0, gid, w).clamp(min=1e-3)
            h = h + self.moe_ctx(torch.einsum("nk,nkld->nld", w, (num / den[:, :, None, None])[gid]))   # 同类的平均编码
            y = self.moe_n(h)
            z = torch.nn.functional.gelu(torch.einsum("nld,kde->nlke", y, self.moe_w1.to(y.dtype)) + self.moe_b1.to(y.dtype))
            z = torch.einsum("nlke,kef->nlkf", z, self.moe_w2.to(y.dtype)) + self.moe_b2.to(y.dtype)
            h = h + (z * w[:, None, :, None].to(z.dtype)).sum(2)                # 各专家按权重合成
        if self.aux_head is not None:
            self.last_aux = self.aux_head(h).squeeze(-1)                       # (N,L):辅助预测,只在训练时用
        return self.head(h).squeeze(-1)


def aux_target(P):
    """辅助目标:下一窗口成交量相对此前 20 日平均半小时成交量的对数,再做截面正态分位。没有 volume 时返回 None。"""
    if P.get("V") is None:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.asarray(P["vs"], np.float64)[:, None, :] * 30.0
        V = np.asarray(P["V"], np.float64)
        A = np.where((V > 0) & (scale > 0), np.log(V) - np.log(scale), np.nan)
    return gauss_rank(A.astype(np.float32), np.asarray(P["M"]))


def forward_days(model, P, d, device, inputs=None):
    """d:(B,) 天下标 -> 分数 (B,K,I);没有行情的 (日,股票) 为 0。
    inputs:已经构建好的 make_inputs 结果(推理时几个模型共用同一份输入,不必重复构建)。"""
    read = torch.as_tensor(P["read"], device=device)
    if not bool(P["insec"][d].any()):                           # 这几天截面表里没有任何股票
        return torch.zeros(len(d), len(read), P["M"].shape[2], device=device)
    x, hist, hmask, b_idx, i_idx = inputs if inputs is not None else make_inputs(P, d, device, model.hist_days)
    x = x[:, :model.n_auc + int(read.max()) * model.seg]        # 最后一个截面之后的 bar 当天用不到,不必编码
    out = model(x, b_idx.to(device), len(d), hist, hmask)[:, read]   # (N,K)
    S = torch.zeros(len(d), P["M"].shape[2], out.shape[1], dtype=out.dtype, device=device)
    S[b_idx.to(device), i_idx.to(device)] = out
    return S.permute(0, 2, 1)


def fit_one(P, Yz, Mz, cfg, seed, device, steps=STEPS, ir=IR_PENALTY, ac=TURN_PENALTY, amp=True, aux=None, aux_w=AUX_WEIGHT):
    """在面板 P(to_tensors 的结果)的 [in_range] 天上训练一个模型。Yz/Mz:标签的正态分位与有效掩码。
    aux = (Az, Am):辅助目标及其掩码(张量);为 None 或模型没有辅助输出头时只用主目标。"""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = RawFormer(**cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.05)
    days = np.flatnonzero(np.asarray(P["in_range"]) & (Mz.sum(-1) >= MIN_NAMES).any(1).cpu().numpy())
    amp = amp and device.type == "cuda"
    model.train()
    t0, acc = time.time(), 0.0
    for it in range(1, steps + 1):
        d = torch.as_tensor(np.sort(rng.choice(days, min(BATCH_DAYS, len(days)), replace=False)), device=P["M"].device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            S = forward_days(model, P, d, device)
        S = S.float()
        y, m, M = Yz[d].to(device), Mz[d].to(device), P["M"][d].to(device)
        ic = xs_corr(S, y, m)
        use = m.sum(-1) >= MIN_NAMES
        loss = -(ic[use]).mean() + (ir * ic[use].std() if ir > 0 else 0.0)
        if ac > 0:                                              # 相邻截面分数越像,换手越低
            loss = loss - ac * xs_corr(S[:, 1:], S[:, :-1], M[:, 1:] & M[:, :-1]).mean()
        if aux is not None and model.aux_head is not None:      # 辅助目标:下一窗口的异常成交量
            b_idx, i_idx = torch.nonzero(P["insec"][d], as_tuple=True)          # 与 make_inputs 的顺序相同
            la = model.last_aux.float()[:, torch.as_tensor(P["read"], device=device)]
            A = torch.zeros(len(d), P["M"].shape[2], la.shape[1], device=device)
            A[b_idx.to(device), i_idx.to(device)] = la
            ya, ma = aux[0][d].to(device), aux[1][d].to(device)
            ica = xs_corr(A.permute(0, 2, 1), ya, ma)
            ua = ma.sum(-1) >= MIN_NAMES
            if bool(ua.any()):
                loss = loss - aux_w * ica[ua].mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        acc += float(ic[use].mean().detach())
        if it % 500 == 0 or it == steps:
            logger.info("训练中", seed=seed, step=it, train_ic=round(acc / (500 if it % 500 == 0 else it % 500), 4),
                        elapsed=round(time.time() - t0, 1))
            acc = 0.0
    return model.eval()


# ---------- 打分(训练与推理共用) ----------
def _zscore(S, M):
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Sm = np.where(M, S, np.nan)
        mu, sd = np.nanmean(Sm, -1, keepdims=True), np.nanstd(Sm, -1, keepdims=True)
        return np.nan_to_num((Sm - mu) / np.where(sd > 1e-12, sd, np.inf))

def score_panel(models, P, ctx_stats, device, smooth=SMOOTH, neutralize=False, batch_days=4):
    """面板 -> 最终分数 (D,K,I):多模型平均 -> 截面标准化 -> [截面间平滑] -> 截面标准化。
    缓冲段(in_range 为 False 的天)不打分。

    ctx_stats / neutralize 在端到端版本里不使用;保留这两个参数只是为了让 test.py 在各个版本之间
    一字不改(test.py 是固定模板,版本差异全部收在 train.py 里)。"""
    t0 = time.time()
    limit_threads()
    T = to_tensors(P)
    need = np.flatnonzero(np.asarray(P["in_range"]))
    out = np.zeros(P["M"].shape, np.float64)
    with torch.no_grad():
        for s in range(0, len(need), batch_days):
            d = torch.as_tensor(need[s:s + batch_days])
            acc = 0.0
            inputs = make_inputs(T, d, device, models[0].hist_days) if bool(T["insec"][d].any()) else None   # 各模型共用
            for m in models:
                acc = acc + forward_days(m, T, d, device, inputs).float()
            out[need[s:s + batch_days]] = (acc / len(models)).double().cpu().numpy()
    Mn = np.asarray(P["M"])
    S = _zscore(out, Mn)
    if smooth > 0:
        D, K, I = S.shape
        flat, prev = S.reshape(D * K, I), np.zeros(I)
        for t in range(D * K):                                  # 按时间顺序递推,只用过去
            prev = (1 - smooth) * flat[t] + smooth * prev
            flat[t] = prev
        S = _zscore(flat.reshape(D, K, I), Mn)
    logger.info("打分完成", days=int(len(need)), n_models=len(models), elapsed=round(time.time() - t0, 1))
    return S


def scores_to_frame(S, P, sections, col="factor"):
    """把 (D,K,I) 分数按截面表的 (date, instrument) 逐行取出;取不到的行填 0(截面中位)。"""
    d = np.searchsorted(P["days"], sections["date"].to_numpy("datetime64[D]"))
    k = np.searchsorted(P["grid"], minute_index(sections["date"]))
    ins = sections["instrument"].to_numpy()
    i = np.searchsorted(P["insts"], ins)
    nD, K, nI = S.shape
    ok = (d < nD) & (k < K) & (i < nI)
    ok &= P["insts"][np.clip(i, 0, nI - 1)] == ins
    val = np.zeros(len(sections))
    val[ok] = S[d[ok], k[ok], i[ok]]
    out = sections[["date", "instrument"]].copy()
    out[col] = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
    return out


# ==== 模型存/读:一律用文本类文件(JSON),不使用 .pt 等二进制 ====
FORMAT = "jy_rawformer_v3"


def _pack(sd):
    return {k: {"dtype": str(v.dtype).replace("torch.", ""), "shape": list(v.shape),
                "data": [float("%.5g" % x) for x in v.detach().cpu().reshape(-1).tolist()]}   # 5 位有效数字,控制文件体积
            for k, v in sd.items()}


def save_model(ckpt, model_path=MODEL_PATH):
    """把 checkpoint 存成 JSON:每个张量转成 {dtype, shape, data(扁平 list)},其余字段原样写入。"""
    payload = {k: v for k, v in ckpt.items() if k != "state_dicts"}
    payload["state_dicts"] = [_pack(sd) for sd in ckpt["state_dicts"]]
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return model_path


def load_model(model_path=MODEL_PATH, map_location="cpu"):
    """读取 save_model 写出的 JSON,把每个 state_dict 还原为 {name: Tensor}。"""
    with open(model_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    ckpt = {k: v for k, v in payload.items() if k != "state_dicts"}
    ckpt.setdefault("ctx_mean", [])                             # test.py 模板会读这两个键;端到端版本不用
    ckpt.setdefault("ctx_std", [])
    ckpt["state_dicts"] = [
        {k: torch.tensor(m["data"], dtype=getattr(torch, m["dtype"])).reshape(m["shape"]).to(map_location)
         for k, m in sd.items()} for sd in payload["state_dicts"]]
    return ckpt


def build_models(ckpt, device):
    if ckpt.get("format") != FORMAT or list(ckpt["feature_names"]) != BAR_COLS:
        raise RuntimeError("model.json 与当前 train.py 不匹配,请重新训练")
    models = []
    for sd in ckpt["state_dicts"]:
        m = RawFormer(**ckpt["model_cfg"]).to(device)
        m.load_state_dict(sd)
        models.append(m.eval())
    return models


# ==== 训练并持久化 ====
def train_and_save(datasources, model_path=MODEL_PATH):
    """在写死的训练区间上从零训练,把 权重 + 结构超参 + 预处理参数 一并存盘。"""
    table = datasources["bar1m"]
    np.random.seed(SEEDS[0])
    torch.manual_seed(SEEDS[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("训练设备", device=str(device), table=table, start=TRAIN_START, end=TRAIN_END,cpu_threads=limit_threads())

    P = build_panel(table, TRAIN_START, TRAIN_END, with_labels=True)
    Yz, Mz = gauss_rank(P["Y"], P["M"])
    if not Mz.any():
        raise RuntimeError("训练区间内没有可用标签")
    aux = aux_target(P)
    aux = None if aux is None else (torch.from_numpy(aux[0]), torch.from_numpy(aux[1]))
    logger.info("辅助目标", used=aux is not None)
    T = to_tensors(P)
    Yz, Mz = torch.from_numpy(Yz), torch.from_numpy(Mz)
    if device.type == "cuda":                                   # 原始面板显存放得下就常驻 GPU,否则留在内存按批搬运
        big = P["G"].nbytes + 4e9 + 5e8 * HIST_DAYS > torch.cuda.mem_get_info()[0]   # 给训练本身和历史输入留出显存
        T = to_tensors(P, device, keep_cpu=("G",) if big else ())
        Yz, Mz = Yz.to(device), Mz.to(device)
        aux = None if aux is None else (aux[0].to(device), aux[1].to(device))

    models = [fit_one(T, Yz, Mz, MODEL_CFG, seed, device, steps=STEPS, aux=aux) for seed in SEEDS]
    logger.info("可训练参数量", n_params=sum(p.numel() for p in models[0].parameters()), n_models=len(models))

    save_model({
        "format": FORMAT,
        "state_dicts": [m.state_dict() for m in models],
        "model_cfg": MODEL_CFG,
        "feature_names": BAR_COLS,                              # 端到端:模型输入就是这 25 个原始字段(adjust_factor 只用于对齐)
        "ctx_mean": [], "ctx_std": [],                          # 与 v1 的文件结构保持一致;端到端版本不用
        "post": {"smooth": SMOOTH, "neutralize": False, "style_feats": [],
                 "prep": {"n_auc": N_AUC, "n_pos": N_POS, "vol_win": VOL_WIN, "price_clip": PRICE_CLIP,
                          "vol_clip": VOL_CLIP}},
        "data_alignment": DATA_ALIGNMENT,
        "train": {"start": TRAIN_START, "end": TRAIN_END, "seeds": list(SEEDS), "steps": STEPS,
                  "label_table": LABEL_TABLE, "section_grid": [int(g) for g in P["grid"]],
                  "n_days": int(P["in_range"].sum()), "n_labels": int(Mz.sum()),
                  "ir_penalty": IR_PENALTY, "turn_penalty": TURN_PENALTY, "aux_weight": AUX_WEIGHT},
    }, model_path)
    logger.info("模型已保存,请随 test.py 一并上传", path=model_path)
    return model_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the archived v12_new configuration from scratch")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--bar-table", default="cpt_jyc_2026_stock_bar1m")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.model_path)), exist_ok=True)
    train_and_save({"bar1m": args.bar_table}, args.model_path)
