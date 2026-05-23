"""
mpeg_codec.py
=============
Simplified MPEG-4-like video codec.
Multimedia Systems - M1 IL G3 - USTHB 2025/2026
BENLALAM Mohamed Rayane  222231363816
AKKOUCHE Mehdi           222231370206

Pipeline :
  encode(frames, params) -> bytes
    BGR -> YCbCr -> 4:2:0 -> I-frames (DCT + quant + zigzag + RLE + Huffman)
                          -> P-frames (motion + residual + zigzag + RLE + Huffman)
    -> packed bitstream -> .bin

  decode(blob) -> [frames]
    inverse de chaque etape ci-dessus
"""

import struct
from collections import namedtuple, Counter
import heapq
import cv2
import numpy as np

# ──────────────────────────────────────────────────────────
# PARAMETRES
# ──────────────────────────────────────────────────────────

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

# Ordre zigzag standard 8x8
_ZIGZAG_IDX = [
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
    (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
    (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
    (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
    (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7),
]


def make_qtables(quality):
    """Construit les matrices de quantification luma/chroma pour un facteur de qualite donne."""
    q = max(1, min(100, int(quality)))
    s = (5000.0 / q) if q < 50 else (200.0 - 2.0 * q)
    def _scale(t):
        return np.clip(np.floor((t * s + 50.0) / 100.0), 1, 255).astype(np.int32)
    return _scale(_BASE_Q_Y), _scale(_BASE_Q_C)


# ──────────────────────────────────────────────────────────
# PARTIE 1 : CONVERSION COULEUR + SOUS-ECHANTILLONNAGE
# ──────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────
# PARTIE 2 : ZIGZAG + RLE + HUFFMAN
# ──────────────────────────────────────────────────────────

def zigzag_scan(block):
    """
    Lecture en zigzag d'un bloc 8x8.
    Regroupe les coefficients en partant du coin haut-gauche (DC)
    vers le coin bas-droite (hautes frequences), en diagonale.
    Les zeros (hautes frequences apres quantification) se retrouvent
    ainsi groupes a la fin — ideal pour le RLE.
    """
    return [int(block[r, c]) for r, c in _ZIGZAG_IDX]


def zigzag_unscan(flat):
    """Inverse du zigzag : remet les 64 valeurs dans le bloc 8x8."""
    block = np.zeros((8, 8), dtype=np.int16)
    for val, (r, c) in zip(flat, _ZIGZAG_IDX):
        block[r, c] = val
    return block


def rle_encode(seq):
    """
    Codage RLE sur une sequence de coefficients zigzag.
    Format : liste de tuples (zero_run, value)
      - zero_run = nombre de zeros consecutifs avant value
      - value    = coefficient non nul (ou 0 pour EOB)
    Le tuple (0, 0) = EOB (End Of Block) signale la fin des coefficients non nuls.
    """
    result = []
    zero_run = 0
    for val in seq:
        if val == 0:
            zero_run += 1
        else:
            result.append((zero_run, val))
            zero_run = 0
    result.append((0, 0))  # EOB
    return result


def rle_decode(pairs, length=64):
    """
    Decodage RLE : reconstruit la sequence de 64 coefficients.
    S'arrete au symbole EOB (0, 0).
    """
    seq = []
    for zero_run, val in pairs:
        if zero_run == 0 and val == 0:  # EOB
            break
        seq.extend([0] * zero_run)
        seq.append(val)
    # Completer avec des zeros jusqu'a 64
    seq.extend([0] * (length - len(seq)))
    return seq[:length]


# ── Huffman ────────────────────────────────────────────────

class HuffmanNode:
    """Noeud de l'arbre de Huffman."""
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq   = freq
        self.left   = None
        self.right  = None
    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(symbols):
    """
    Construit l'arbre de Huffman a partir d'une liste de symboles.
    Les symboles les plus frequents recoivent les codes les plus courts.
    """
    freq = Counter(symbols)
    heap = [HuffmanNode(s, f) for s, f in freq.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        node = heapq.heappop(heap)
        root = HuffmanNode(None, node.freq)
        root.left = node
        return root

    while len(heap) > 1:
        left  = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = HuffmanNode(None, left.freq + right.freq)
        parent.left  = left
        parent.right = right
        heapq.heappush(heap, parent)

    return heap[0]


def build_codebook(node, prefix="", codebook=None):
    """Parcourt l'arbre et assigne un code binaire a chaque symbole."""
    if codebook is None:
        codebook = {}
    if node is None:
        return codebook
    if node.symbol is not None:
        codebook[node.symbol] = prefix if prefix else "0"
        return codebook
    build_codebook(node.left,  prefix + "0", codebook)
    build_codebook(node.right, prefix + "1", codebook)
    return codebook


def huffman_encode(symbols, codebook):
    """
    Encode une liste de symboles en une chaine de bits.
    Retourne la chaine de bits + le nombre de bits utiles (pour le padding).
    """
    bitstring = "".join(codebook[s] for s in symbols)
    # Padding pour avoir un nombre entier d'octets
    pad = (8 - len(bitstring) % 8) % 8
    bitstring += "0" * pad
    # Convertir en bytes
    data = bytes(int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8))
    return data, pad


def huffman_decode(data, pad, codebook, n_symbols):
    """
    Decode des bytes en symboles a partir du codebook.
    Utilise le codebook inverse (bits -> symbole).
    """
    # Reconstruire le codebook inverse
    inv = {v: k for k, v in codebook.items()}
    # Convertir les bytes en bitstring
    bitstring = "".join(f"{b:08b}" for b in data)
    if pad > 0:
        bitstring = bitstring[:-pad]

    symbols = []
    current = ""
    for bit in bitstring:
        current += bit
        if current in inv:
            symbols.append(inv[current])
            current = ""
            if len(symbols) == n_symbols:
                break
    return symbols


def serialize_codebook(codebook):
    """Serialise le codebook en bytes pour le stocker dans le bitstream."""
    # Format : nb_entrees | pour chaque entree : symbol(4o) + len_code(1o) + code_bits
    entries = []
    for symbol, code in codebook.items():
        # symbol est un tuple (zero_run, value) ou un int
        sym_bytes = struct.pack("<ii", *symbol) if isinstance(symbol, tuple) else struct.pack("<i", symbol)
        code_len  = len(code)
        pad = (8 - code_len % 8) % 8
        code_padded = code + "0" * pad
        code_bytes  = bytes(int(code_padded[i:i+8], 2) for i in range(0, len(code_padded), 8))
        entries.append(struct.pack("<B", len(sym_bytes)) + sym_bytes +
                       struct.pack("<BB", code_len, pad) + code_bytes)
    header = struct.pack("<H", len(entries))
    return header + b"".join(entries)


def deserialize_codebook(buf, offset):
    """Deserialise un codebook depuis le buffer."""
    n = struct.unpack_from("<H", buf, offset)[0]; offset += 2
    codebook = {}
    for _ in range(n):
        sym_len = buf[offset]; offset += 1
        if sym_len == 8:
            zero_run, value = struct.unpack_from("<ii", buf, offset)
            symbol = (zero_run, value)
        else:
            symbol = struct.unpack_from("<i", buf, offset)[0]
        offset += sym_len
        code_len, pad = buf[offset], buf[offset+1]; offset += 2
        nb = (code_len + 7) // 8
        code_bytes = buf[offset:offset+nb]; offset += nb
        bitstring = "".join(f"{b:08b}" for b in code_bytes)
        code = bitstring[:code_len]
        codebook[symbol] = code
    return codebook, offset


# ──────────────────────────────────────────────────────────
# PARTIE 2 : CODAGE INTRA (I-FRAMES) - DCT + QUANT + ZIGZAG + RLE + HUFFMAN
# ──────────────────────────────────────────────────────────

def encode_plane_intra(plane, q_table, block=8):
    """
    Encode un canal complet (Y, Cb ou Cr) pour un I-frame :
    1. Centre autour de 0 (-128)
    2. DCT 8x8 sur chaque bloc
    3. Quantification par q_table
    4. Lecture en zigzag
    5. RLE sur les coefficients zigzag
    6. Huffman sur les paires RLE
    Retourne les bytes encodes + codebook + dimensions.
    """
    centred = plane.astype(np.float32) - 128.0
    h, w    = centred.shape
    all_pairs = []  # toutes les paires RLE de tous les blocs

    # Etape 1-4 : DCT + quant + zigzag + RLE
    blocks_rle = []
    for by in range(h // block):
        for bx in range(w // block):
            tile    = centred[by*block:(by+1)*block, bx*block:(bx+1)*block]
            dct_c   = cv2.dct(tile)
            quant   = np.round(dct_c / q_table).astype(np.int16)
            zz      = zigzag_scan(quant)
            pairs   = rle_encode(zz)
            blocks_rle.append(pairs)
            all_pairs.extend(pairs)

    # Etape 5 : construire le codebook Huffman sur toutes les paires
    tree     = build_huffman_tree(all_pairs)
    codebook = build_codebook(tree)

    # Etape 6 : encoder avec Huffman
    flat_pairs = [p for blk in blocks_rle for p in blk]
    data, pad  = huffman_encode(flat_pairs, codebook)

    # Stocker : nb_blocs_h, nb_blocs_w, codebook, pad, data
    header = struct.pack("<HHB", h // block, w // block, pad)
    cb_bytes = serialize_codebook(codebook)
    data_len = struct.pack("<I", len(data))
    return header + cb_bytes + data_len + data


def decode_plane_intra(buf, offset, q_table, block=8):
    """
    Decode un canal I-frame : inverse exact de encode_plane_intra.
    1. Huffman decode
    2. RLE decode
    3. Zigzag inverse
    4. Dequantification
    5. IDCT
    6. Recentre (+128)
    """
    rows, cols, pad = struct.unpack_from("<HHB", buf, offset); offset += 5
    codebook, offset = deserialize_codebook(buf, offset)
    data_len = struct.unpack_from("<I", buf, offset)[0]; offset += 4
    data     = buf[offset:offset+data_len]; offset += data_len

    n_blocks    = rows * cols
    # Decoder toutes les paires Huffman
    # Nombre de paires = inconnu a l'avance, on decode bloc par bloc
    inv_cb  = {v: k for k, v in codebook.items()}
    bitstring = "".join(f"{b:08b}" for b in data)
    if pad > 0:
        bitstring = bitstring[:-pad]

    h_out = rows * block
    w_out = cols * block
    plane = np.zeros((h_out, w_out), dtype=np.float32)

    bit_idx = 0
    for by in range(rows):
        for bx in range(cols):
            # Decoder les paires RLE de ce bloc jusqu'a EOB
            pairs = []
            current = ""
            while True:
                if bit_idx >= len(bitstring):
                    break
                current += bitstring[bit_idx]; bit_idx += 1
                if current in inv_cb:
                    sym = inv_cb[current]
                    pairs.append(sym)
                    current = ""
                    if sym == (0, 0):  # EOB
                        break

            seq   = rle_decode(pairs)
            block_2d = zigzag_unscan(seq)
            dequant  = block_2d.astype(np.float32) * q_table
            recon    = cv2.idct(dequant) + 128.0
            plane[by*block:(by+1)*block, bx*block:(bx+1)*block] = recon

    return np.clip(plane, 0, 255), offset


def encode_plane_residual(residual, q_table, block=8):
    """Encode un residu de P-frame (pas de recentrage)."""
    h, w = residual.shape
    all_pairs = []
    blocks_rle = []

    for by in range(h // block):
        for bx in range(w // block):
            tile  = residual[by*block:(by+1)*block, bx*block:(bx+1)*block].astype(np.float32)
            dct_c = cv2.dct(tile)
            quant = np.round(dct_c / q_table).astype(np.int16)
            zz    = zigzag_scan(quant)
            pairs = rle_encode(zz)
            blocks_rle.append(pairs)
            all_pairs.extend(pairs)

    tree     = build_huffman_tree(all_pairs)
    codebook = build_codebook(tree)
    flat_pairs = [p for blk in blocks_rle for p in blk]
    data, pad  = huffman_encode(flat_pairs, codebook)

    header   = struct.pack("<HHB", h // block, w // block, pad)
    cb_bytes = serialize_codebook(codebook)
    data_len = struct.pack("<I", len(data))
    return header + cb_bytes + data_len + data


def decode_plane_residual(buf, offset, q_table, block=8):
    """Decode un residu de P-frame."""
    rows, cols, pad = struct.unpack_from("<HHB", buf, offset); offset += 5
    codebook, offset = deserialize_codebook(buf, offset)
    data_len = struct.unpack_from("<I", buf, offset)[0]; offset += 4
    data     = buf[offset:offset+data_len]; offset += data_len

    inv_cb    = {v: k for k, v in codebook.items()}
    bitstring = "".join(f"{b:08b}" for b in data)
    if pad > 0:
        bitstring = bitstring[:-pad]

    h_out = rows * block
    w_out = cols * block
    plane = np.zeros((h_out, w_out), dtype=np.float32)

    bit_idx = 0
    for by in range(rows):
        for bx in range(cols):
            pairs   = []
            current = ""
            while True:
                if bit_idx >= len(bitstring):
                    break
                current += bitstring[bit_idx]; bit_idx += 1
                if current in inv_cb:
                    sym = inv_cb[current]
                    pairs.append(sym)
                    current = ""
                    if sym == (0, 0):
                        break
            seq      = rle_decode(pairs)
            block_2d = zigzag_unscan(seq)
            dequant  = block_2d.astype(np.float32) * q_table
            recon    = cv2.idct(dequant)
            plane[by*block:(by+1)*block, bx*block:(bx+1)*block] = recon

    return plane, offset


# ──────────────────────────────────────────────────────────
# PARTIE 3 : CODAGE INTER (P-FRAMES) - ESTIMATION DE MOUVEMENT
# ──────────────────────────────────────────────────────────

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
    Complexite : O(log search) au lieu de O(search^2).
    """
    h, w    = current.shape
    rows, cols = h // mb, w // mb
    pad     = search
    ref_pad = np.pad(reference, pad, mode="edge")
    vectors = np.zeros((rows, cols, 2), dtype=np.int16)

    for by in range(rows):
        for bx in range(cols):
            y0, x0 = by * mb, bx * mb
            block  = current[y0:y0+mb, x0:x0+mb]
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
    h, w       = reference.shape
    rows, cols, _ = vectors.shape
    pad        = int(np.max(np.abs(vectors))) if vectors.size else 0
    ref_pad    = np.pad(reference, pad, mode="edge")
    pred       = np.zeros_like(reference)

    for by in range(rows):
        for bx in range(cols):
            dy, dx = vectors[by, bx]
            y0, x0 = by * mb, bx * mb
            pred[y0:y0+mb, x0:x0+mb] = ref_pad[
                y0+dy+pad:y0+dy+pad+mb,
                x0+dx+pad:x0+dx+pad+mb
            ]
    return pred


def pack_motion_vectors(vectors):
    """Serialise les vecteurs de mouvement en bytes."""
    shape = vectors.shape
    return struct.pack("<HH", shape[0], shape[1]) + vectors.tobytes()


def unpack_motion_vectors(buf, offset):
    """Deserialise les vecteurs de mouvement."""
    rows, cols = struct.unpack_from("<HH", buf, offset); offset += 4
    nb = rows * cols * 2
    arr = np.frombuffer(buf, dtype=np.int16, count=nb, offset=offset).reshape(rows, cols, 2).copy()
    offset += nb * 2
    return arr, offset


# ──────────────────────────────────────────────────────────
# PARTIE 4 : BITSTREAM
# ──────────────────────────────────────────────────────────

_MAGIC   = b"SM42"
_VERSION = b"\x02"


def pack_bitstream(params, luma_shape, chroma_shape, records):
    """Assemble le bitstream complet."""
    parts = [_MAGIC, _VERSION]
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
            parts += [b"I", rec["y"], rec["cb"], rec["cr"]]
        else:
            parts += [b"P",
                      pack_motion_vectors(rec["mv"]),
                      rec["y"], rec["cb"], rec["cr"]]
    return b"".join(parts)


def unpack_bitstream(blob):
    """Lit et parse le bitstream."""
    if blob[:4] != _MAGIC:
        raise ValueError("Bitstream invalide")
    offset = 5
    (n, ly, lx, cy, cx, gop, q, mb, search, sub) = struct.unpack_from(
        "<H HH HH B B B B B", blob, offset)
    offset += struct.calcsize("<H HH HH B B B B B")
    params = Params(gop=gop, quality=q, block=8,
                    macroblock=mb, search=search, subsample=bool(sub))
    return params, (ly, lx), (cy, cx), offset, n, blob


# ──────────────────────────────────────────────────────────
# FONCTIONS PRINCIPALES : ENCODE / DECODE
# ──────────────────────────────────────────────────────────

def encode(frames_bgr, params=DEFAULT_PARAMS):
    """
    Encode une liste de frames BGR en un bitstream compresse.
    Pipeline complet : BGR->YCbCr->4:2:0 -> DCT->Quant->Zigzag->RLE->Huffman
    """
    q_y, q_c = make_qtables(params.quality)
    mb, block = params.macroblock, params.block
    luma_shape = chroma_shape = None
    records = []
    prev = None

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
            # ── I-FRAME ──
            ey  = encode_plane_intra(y,  q_y, block)
            ecb = encode_plane_intra(cb, q_c, block)
            ecr = encode_plane_intra(cr, q_c, block)
            records.append({"type": "I", "y": ey, "cb": ecb, "cr": ecr})

            # Reconstruire pour reference
            ry,  _ = decode_plane_intra(ey,  0, q_y, block)
            rcb, _ = decode_plane_intra(ecb, 0, q_c, block)
            rcr, _ = decode_plane_intra(ecr, 0, q_c, block)

        else:
            # ── P-FRAME ──
            py, pcb, pcr = prev
            mv = three_step_search(y.astype(np.uint8), py.astype(np.uint8),
                                   mb, params.search)

            pred_y   = motion_compensate(py, mv, mb)
            res_y    = y.astype(np.float32) - pred_y.astype(np.float32)
            ey       = encode_plane_residual(res_y, q_y, block)
            recon_ry, _ = decode_plane_residual(ey, 0, q_y, block)
            ry       = np.clip(pred_y + recon_ry, 0, 255)

            mv_c  = (mv // 2).astype(np.int16) if params.subsample else mv
            mb_c  = mb // 2 if params.subsample else mb
            pred_cb = motion_compensate(pcb, mv_c, mb_c)
            pred_cr = motion_compensate(pcr, mv_c, mb_c)
            res_cb  = cb.astype(np.float32) - pred_cb
            res_cr  = cr.astype(np.float32) - pred_cr
            ecb = encode_plane_residual(res_cb, q_c, block)
            ecr = encode_plane_residual(res_cr, q_c, block)
            rcb_r, _ = decode_plane_residual(ecb, 0, q_c, block)
            rcr_r, _ = decode_plane_residual(ecr, 0, q_c, block)
            rcb = np.clip(pred_cb + rcb_r, 0, 255)
            rcr = np.clip(pred_cr + rcr_r, 0, 255)

            records.append({"type": "P", "mv": mv, "y": ey, "cb": ecb, "cr": ecr})

        prev = (ry, rcb, rcr)

    return pack_bitstream(params, luma_shape, chroma_shape, records)


def decode(blob, output_shape=None):
    """
    Decode un bitstream et retourne la liste des frames BGR reconstruites.
    Pipeline inverse : Huffman->RLE->Zigzag->Dequant->IDCT->YCbCr->BGR
    """
    params, luma_shape, chroma_shape, offset, n, buf = unpack_bitstream(blob)
    q_y, q_c = make_qtables(params.quality)
    mb, block = params.macroblock, params.block
    out_frames = []
    prev = None

    for _ in range(n):
        tag = chr(buf[offset]); offset += 1

        if tag == "I":
            y,  offset = decode_plane_intra(buf, offset, q_y, block)
            cb, offset = decode_plane_intra(buf, offset, q_c, block)
            cr, offset = decode_plane_intra(buf, offset, q_c, block)
        else:
            if prev is None:
                raise RuntimeError("P-frame avant un I-frame")
            mv, offset = unpack_motion_vectors(buf, offset)
            py, pcb, pcr = prev

            res_y,  offset = decode_plane_residual(buf, offset, q_y, block)
            pred_y  = motion_compensate(py, mv, mb)
            y       = np.clip(pred_y + res_y, 0, 255)

            mv_c  = (mv // 2).astype(np.int16) if params.subsample else mv
            mb_c  = mb // 2 if params.subsample else mb
            res_cb, offset = decode_plane_residual(buf, offset, q_c, block)
            res_cr, offset = decode_plane_residual(buf, offset, q_c, block)
            pred_cb = motion_compensate(pcb, mv_c, mb_c)
            pred_cr = motion_compensate(pcr, mv_c, mb_c)
            cb = np.clip(pred_cb + res_cb, 0, 255)
            cr = np.clip(pred_cr + res_cr, 0, 255)

        prev = (y, cb, cr)

        cb_full = chroma_up(cb, y.shape) if params.subsample else cb
        cr_full = chroma_up(cr, y.shape) if params.subsample else cr
        ycbcr   = np.stack([y, cb_full, cr_full], axis=-1)
        bgr     = ycbcr_to_bgr(ycbcr)

        if output_shape is not None:
            h, w = output_shape
            bgr  = bgr[:h, :w]
        out_frames.append(bgr)

    return out_frames, params, []  # records vide pour compat viz.py


# ──────────────────────────────────────────────────────────
# PARTIE 5a : METRIQUES
# ──────────────────────────────────────────────────────────

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
