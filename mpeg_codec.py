import bz2
import struct
from collections import namedtuple
import cv2
import numpy as np

#les parametres

Params = namedtuple("Params", "gop quality block macroblock search subsample")

DEFAULT_PARAMS = Params(gop=8, quality=50, block=8,
                        macroblock=16, search=8, subsample=True)

# Matrices de quantification standard JPEG (luma + chroma)
_BASE_Q_Y = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99],
], dtype=np.float32)

_BASE_Q_C = np.array([
    [17,18,24,47,99,99,99,99],
    [18,21,26,66,99,99,99,99],
    [24,26,56,99,99,99,99,99],
    [47,66,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
], dtype=np.float32)


def make_qtables(quality):
    """Construit les matrices de quantification luma/chroma pour un facteur de qualite donne."""
    q = max(1, min(100, int(quality)))
    s = (5000.0 / q) if q < 50 else (200.0 - 2.0 * q)
    def _scale(t):
        return np.clip(np.floor((t * s + 50.0) / 100.0), 1, 255).astype(np.int32)
    return _scale(_BASE_Q_Y), _scale(_BASE_Q_C)


#partie 1 : conversion couleurs et sous échantillonnage

def bgr_to_ycbcr(bgr):
    """BGR -> YCbCr (BT.601 via OpenCV)."""
    yc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    return yc[..., [0, 2, 1]]  # reordre : YCrCb -> YCbCr


def ycbcr_to_bgr(ycbcr):
    """YCbCr -> BGR."""
    yc = ycbcr[..., [0, 2, 1]]
    yc = np.clip(yc, 0, 255).astype(np.uint8)
    return cv2.cvtColor(yc, cv2.COLOR_YCrCb2BGR)


def chroma_down(plane):
    """Sous-echantillonnage 4:2:0 : divise par 2 en H et V (filtre boite 2x2)."""
    h, w = plane.shape
    h2, w2 = (h // 2) * 2, (w // 2) * 2
    p = plane[:h2, :w2]
    return p.reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3))


def chroma_up(plane, target_hw):
    """Sur-echantillonnage : double les dimensions par repetition de pixels."""
    th, tw = target_hw
    up = np.repeat(np.repeat(plane, 2, axis=0), 2, axis=1)
    return up[:th, :tw]


def _pad_to(plane, m):
    """Ajoute du padding pour que les dimensions soient multiples de m."""
    h, w = plane.shape
    ph = (-h) % m
    pw = (-w) % m
    return np.pad(plane, ((0, ph), (0, pw)), mode="edge")


#partie 2 : codage intra (I-frames) - DCT + quantification

def dct_quant_plane(plane, q_table, block):
    """Applique la DCT 8x8 puis la quantification sur toute une image."""
    centred = plane.astype(np.float32) - 128.0  # centrer autour de 0
    return _block_dct_quant(centred, q_table, block)


def dct_quant_residual(residual, q_table, block):
    """DCT + quantification sur un residu (deja centre)."""
    return _block_dct_quant(residual.astype(np.float32), q_table, block)


def _block_dct_quant(plane, q_table, block):
    """Decoupage en blocs 8x8, DCT, puis division par la matrice de quantification."""
    h, w = plane.shape
    q_out = np.empty((h // block, w // block, block, block), dtype=np.int16)
    for by in range(h // block):
        for bx in range(w // block):
            tile = plane[by*block:(by+1)*block, bx*block:(bx+1)*block]
            d = cv2.dct(tile)
            q_out[by, bx] = np.round(d / q_table).astype(np.int16)
    return q_out


def idct_dequant_plane(q_blocks, q_table, block, recentre=True):
    """Inverse : dequantification + DCT inverse. Reconstruit l'image."""
    rows, cols = q_blocks.shape[:2]
    h, w = rows * block, cols * block
    out = np.empty((h, w), dtype=np.float32)
    for by in range(rows):
        for bx in range(cols):
            d = q_blocks[by, bx].astype(np.float32) * q_table
            t = cv2.idct(d)
            out[by*block:(by+1)*block, bx*block:(bx+1)*block] = t
    if recentre:
        out += 128.0
    return out


#partie 3 : codage inter (P-frames) - estimation de mouvement

def _sad(a, b):
    """Sum of Absolute Differences entre deux blocs."""
    return int(np.abs(a.astype(np.int32) - b.astype(np.int32)).sum())


def _fetch(ref_pad, y0, x0, mb, pad):
    """Extrait un bloc de la trame de reference paddee."""
    return ref_pad[y0+pad:y0+pad+mb, x0+pad:x0+pad+mb]


def three_step_search(current, reference, mb, search):
    """
    Algorithme Three-Step Search (TSS) pour l'estimation de mouvement.
    Pour chaque macroblock 16x16, trouve le meilleur bloc correspondant
    dans la trame precedente dans une fenetre de +/-search pixels.
    Complexite : O(log search) au lieu de O(search^2) pour la recherche exhaustive.
    """
    h, w = current.shape
    rows, cols = h // mb, w // mb
    pad = search
    ref_pad = np.pad(reference, pad, mode="edge")
    vectors = np.zeros((rows, cols, 2), dtype=np.int16)

    for by in range(rows):
        for bx in range(cols):
            y0, x0 = by * mb, bx * mb
            block = current[y0:y0+mb, x0:x0+mb]
            best_dy, best_dx = 0, 0
            best = _sad(block, _fetch(ref_pad, y0, x0, mb, pad))
            step = max(1, search // 2)

            while step >= 1:
                cy, cx = best_dy, best_dx
                for dy in (cy-step, cy, cy+step):
                    for dx in (cx-step, cx, cx+step):
                        if abs(dy) > search or abs(dx) > search:
                            continue
                        cand = _fetch(ref_pad, y0+dy, x0+dx, mb, pad)
                        c = _sad(block, cand)
                        if c < best:
                            best, best_dy, best_dx = c, dy, dx
                step //= 2

            vectors[by, bx] = (best_dy, best_dx)
    return vectors


def motion_compensate(reference, vectors, mb):
    """Construit la prediction a partir de la trame precedente et des vecteurs."""
    h, w = reference.shape
    rows, cols, _ = vectors.shape
    pad = int(np.max(np.abs(vectors))) if vectors.size else 0
    ref_pad = np.pad(reference, pad, mode="edge")
    pred = np.zeros_like(reference)

    for by in range(rows):
        for bx in range(cols):
            dy, dx = vectors[by, bx]
            y0, x0 = by * mb, bx * mb
            pred[y0:y0+mb, x0:x0+mb] = ref_pad[
                y0+dy+pad:y0+dy+pad+mb,
                x0+dx+pad:x0+dx+pad+mb
            ]
    return pred


#partie 4 : entropie - bitstream + BZ2

_MAGIC = b"MV2\x00"
_DTYPE_TAGS = {np.int8: 0, np.int16: 1, np.int32: 2}
_TAG_DTYPES  = {v: np.dtype(k) for k, v in _DTYPE_TAGS.items()}


def _pack_array(arr):
    """Serialise un tableau numpy : ndim | shape | dtype_tag | donnees brutes."""
    shape = arr.shape
    header = struct.pack("<B", len(shape)) + struct.pack(f"<{len(shape)}I", *shape)
    tag = _DTYPE_TAGS[arr.dtype.type]
    return header + struct.pack("<B", tag) + arr.tobytes()


def _unpack_array(buf, offset):
    """Deserialise un tableau numpy depuis le buffer."""
    ndim = buf[offset]; offset += 1
    shape = struct.unpack_from(f"<{ndim}I", buf, offset); offset += 4 * ndim
    tag = buf[offset]; offset += 1
    dtype = _TAG_DTYPES[tag]
    count = int(np.prod(shape)) if shape else 0
    arr = np.frombuffer(buf, dtype=dtype, count=count, offset=offset).reshape(shape)
    offset += count * dtype.itemsize
    return arr.copy(), offset


def pack_bitstream(params, luma_shape, chroma_shape, records):
    """Assemble et compresse le bitstream avec bz2."""
    parts = [_MAGIC, b"\x01"]
    parts.append(struct.pack(
        "<H HH HH B B B B B",
        len(records),
        luma_shape[0], luma_shape[1],
        chroma_shape[0], chroma_shape[1],
        params.gop, params.quality, params.macroblock,
        params.search, 1 if params.subsample else 0,
    ))
    for rec in records:
        if rec["type"] == "I":
            parts += [b"I", _pack_array(rec["y"]),
                      _pack_array(rec["cb"]), _pack_array(rec["cr"])]
        else:
            parts += [b"P", _pack_array(rec["mv"]), _pack_array(rec["y"]),
                      _pack_array(rec["cb"]), _pack_array(rec["cr"])]
    return bz2.compress(b"".join(parts), compresslevel=9)


def unpack_bitstream(blob):
    """Decompresse et desserialise le bitstream."""
    raw = bz2.decompress(blob)
    if raw[:4] != _MAGIC:
        raise ValueError("Bitstream invalide")
    offset = 5
    (n, ly, lx, cy, cx, gop, q, mb, search, sub) = struct.unpack_from(
        "<H HH HH B B B B B", raw, offset)
    offset += struct.calcsize("<H HH HH B B B B B")
    params = Params(gop=gop, quality=q, block=8,
                    macroblock=mb, search=search, subsample=bool(sub))
    records = []
    for _ in range(n):
        tag = chr(raw[offset]); offset += 1
        if tag == "I":
            y,  offset = _unpack_array(raw, offset)
            cb, offset = _unpack_array(raw, offset)
            cr, offset = _unpack_array(raw, offset)
            records.append({"type": "I", "y": y, "cb": cb, "cr": cr})
        else:
            mv, offset = _unpack_array(raw, offset)
            y,  offset = _unpack_array(raw, offset)
            cb, offset = _unpack_array(raw, offset)
            cr, offset = _unpack_array(raw, offset)
            records.append({"type": "P", "mv": mv, "y": y, "cb": cb, "cr": cr})
    return params, (ly, lx), (cy, cx), records


#fonctions principales : encode et decode

def encode(frames_bgr, params=DEFAULT_PARAMS):
    """
    Encode une liste de frames BGR en un bitstream compresse (bytes).
    Chaque GOP-ieme frame est un I-frame, les autres sont des P-frames.
    """
    q_y, q_c = make_qtables(params.quality)
    mb, block = params.macroblock, params.block
    luma_shape = chroma_shape = None
    records = []
    prev = None  # (Y, Cb, Cr) reconstruits du frame precedente

    for idx, bgr in enumerate(frames_bgr):
        ycbcr = bgr_to_ycbcr(bgr)
        y, cb, cr = ycbcr[...,0], ycbcr[...,1], ycbcr[...,2]

        if params.subsample:
            cb = chroma_down(cb)
            cr = chroma_down(cr)

        y  = _pad_to(y,  mb)
        cb = _pad_to(cb, block)
        cr = _pad_to(cr, block)

        if luma_shape is None:
            luma_shape   = y.shape
            chroma_shape = cb.shape

        is_intra = (idx % params.gop == 0) or (prev is None)

        if is_intra:
            #I-frame : DCT + quantification directe
            qy  = dct_quant_plane(y,  q_y, block)
            qcb = dct_quant_plane(cb, q_c, block)
            qcr = dct_quant_plane(cr, q_c, block)
            records.append({"type": "I", "y": qy, "cb": qcb, "cr": qcr})
            ry  = np.clip(idct_dequant_plane(qy,  q_y, block), 0, 255)
            rcb = np.clip(idct_dequant_plane(qcb, q_c, block), 0, 255)
            rcr = np.clip(idct_dequant_plane(qcr, q_c, block), 0, 255)

        else:
            # P-frame : estimation de mouvement + DCT du residu
            py, pcb, pcr = prev
            mv = three_step_search(y.astype(np.uint8), py.astype(np.uint8),
                                   mb, params.search)

            pred_y   = motion_compensate(py, mv, mb)
            res_y    = y.astype(np.float32) - pred_y.astype(np.float32)
            qy       = dct_quant_residual(res_y, q_y, block)
            recon_ry = idct_dequant_plane(qy, q_y, block, recentre=False)
            ry       = np.clip(pred_y + recon_ry, 0, 255)

            mv_c  = (mv // 2).astype(np.int16) if params.subsample else mv
            mb_c  = mb // 2                     if params.subsample else mb
            pred_cb = motion_compensate(pcb, mv_c, mb_c)
            pred_cr = motion_compensate(pcr, mv_c, mb_c)
            qcb = dct_quant_residual(cb.astype(np.float32) - pred_cb, q_c, block)
            qcr = dct_quant_residual(cr.astype(np.float32) - pred_cr, q_c, block)
            rcb = np.clip(pred_cb + idct_dequant_plane(qcb, q_c, block, recentre=False), 0, 255)
            rcr = np.clip(pred_cr + idct_dequant_plane(qcr, q_c, block, recentre=False), 0, 255)

            records.append({"type": "P", "mv": mv, "y": qy, "cb": qcb, "cr": qcr})

        prev = (ry, rcb, rcr)

    return pack_bitstream(params, luma_shape, chroma_shape, records)


def decode(blob, output_shape=None):
    """
    Decode un bitstream et retourne la liste des frames BGR reconstruites.
    output_shape = (H, W) original (avant padding). Si None, retourne le padding complet.
    """
    params, luma_shape, chroma_shape, records = unpack_bitstream(blob)
    q_y, q_c = make_qtables(params.quality)
    mb, block = params.macroblock, params.block
    out_frames = []
    prev = None

    for rec in records:
        if rec["type"] == "I":
            y  = np.clip(idct_dequant_plane(rec["y"],  q_y, block), 0, 255)
            cb = np.clip(idct_dequant_plane(rec["cb"], q_c, block), 0, 255)
            cr = np.clip(idct_dequant_plane(rec["cr"], q_c, block), 0, 255)
        else:
            if prev is None:
                raise RuntimeError("P-frame avant un I-frame")
            py, pcb, pcr = prev
            mv     = rec["mv"]
            pred_y = motion_compensate(py, mv, mb)
            res_y  = idct_dequant_plane(rec["y"], q_y, block, recentre=False)
            y      = np.clip(pred_y + res_y, 0, 255)

            mv_c  = (mv // 2).astype(np.int16) if params.subsample else mv
            mb_c  = mb // 2                     if params.subsample else mb
            pred_cb = motion_compensate(pcb, mv_c, mb_c)
            pred_cr = motion_compensate(pcr, mv_c, mb_c)
            cb = np.clip(pred_cb + idct_dequant_plane(rec["cb"], q_c, block, recentre=False), 0, 255)
            cr = np.clip(pred_cr + idct_dequant_plane(rec["cr"], q_c, block, recentre=False), 0, 255)

        prev = (y, cb, cr)

        cb_full = chroma_up(cb, y.shape) if params.subsample else cb
        cr_full = chroma_up(cr, y.shape) if params.subsample else cr
        ycbcr   = np.stack([y, cb_full, cr_full], axis=-1)
        bgr     = ycbcr_to_bgr(ycbcr)

        if output_shape is not None:
            h, w = output_shape
            bgr = bgr[:h, :w]
        out_frames.append(bgr)

    return out_frames, params, records


#partie 5a : metriques

def psnr(a, b):
    """Calcule le PSNR en dB entre deux frames."""
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0) - 10.0 * np.log10(mse)


def frame_breakdown(records):
    """Retourne (nb_I_frames, nb_P_frames)."""
    n_i = sum(1 for r in records if r["type"] == "I")
    n_p = sum(1 for r in records if r["type"] == "P")
    return n_i, n_p
