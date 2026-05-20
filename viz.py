"""
viz.py  —  Visualisation du pipeline + courbes d'analyse
BENLALAM Mohamed Rayane  222231363816
AKKOUCHE Mehdi           222231370206
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mpeg_codec import (encode, decode, psnr, frame_breakdown,
                        bgr_to_ycbcr, chroma_down, chroma_up,
                        make_qtables, _pad_to, dct_quant_plane,
                        idct_dequant_plane, Params, DEFAULT_PARAMS)


def _rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def make_pipeline_figure(orig_frames, recon_frames, records, params, out_path):
    """
    Produit une figure unique avec 5 lignes :
      1. Frames originales
      2. Canaux Y / Cb / Cr
      3. Bloc 8x8 : pixels -> DCT -> quantifie -> reconstruit
      4. Vecteurs de mouvement sur une P-frame + carte des residus
      5. Frames reconstruites
    """
    q_y, q_c = make_qtables(params.quality)

    fig = plt.figure(figsize=(18, 20), facecolor="white")
    fig.suptitle("MPEG-4-like pipeline — stage visualisation",
                 fontsize=13, fontweight="bold", y=0.99)

    n_show = min(6, len(orig_frames))

    #Ligne 1 : frames originales
    for i in range(n_show):
        ax = fig.add_subplot(5, n_show, i + 1)
        ax.imshow(_rgb(orig_frames[i]))
        ftype = records[i]["type"] if i < len(records) else "?"
        ax.set_title(f"orig {i}", fontsize=8)
        if i == 0:
            ax.set_ylabel("1] originals", rotation=0, labelpad=55, fontsize=8, va="center")
        ax.axis("off")

    #Ligne 2 : canaux Y, Cb, Cr 
    ycbcr0 = bgr_to_ycbcr(orig_frames[0])
    Y0  = ycbcr0[..., 0]
    Cb0 = chroma_up(chroma_down(ycbcr0[..., 1]), Y0.shape)
    Cr0 = chroma_up(chroma_down(ycbcr0[..., 2]), Y0.shape)

    for col, (ch, name, cmap) in enumerate([
        (Y0, "Y", "gray"),
        (Cb0, "Cb", "Blues"),
        (Cr0, "Cr", "Reds"),
    ]):
        ax = fig.add_subplot(5, n_show, n_show + col + 1)
        ax.imshow(ch, cmap=cmap)
        ax.set_title(name, fontsize=9)
        if col == 0:
            ax.set_ylabel("2] Y / Cb / Cr", rotation=0, labelpad=55, fontsize=8, va="center")
        ax.axis("off")

    #Ligne 3 : bloc 8x8 a travers le pipeline DCT
    gray0 = cv2.cvtColor(orig_frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    #Choisir un bloc representatif (pas le coin en haut a gauche qui est souvent uniforme)
    brow, bcol = 2, 4
    raw_block   = gray0[brow*8:(brow+1)*8, bcol*8:(bcol+1)*8]
    dct_block   = cv2.dct(raw_block - 128.0)
    quant_block = np.round(dct_block / q_y).astype(np.float32)
    recon_block = cv2.idct(quant_block * q_y) + 128.0
    q_table_vis = make_qtables(params.quality)[0]

    dct_stages = [
        (raw_block,         "raw pixels",         "viridis"),
        (np.log1p(np.abs(dct_block)), "DCT",      "hot"),
        (quant_block,       "quantised",           "RdBu"),
        (recon_block,       "reconstructed",       "viridis"),
        (q_table_vis,       f"Q-table (Q={params.quality})", "plasma"),
    ]
    for col, (blk, name, cmap) in enumerate(dct_stages):
        ax = fig.add_subplot(5, n_show, 2*n_show + col + 1)
        im = ax.imshow(blk, cmap=cmap, aspect="auto")
        ax.set_title(name, fontsize=8)
        if col == 0:
            ax.set_ylabel("3] DCT & quant", rotation=0, labelpad=55, fontsize=8, va="center")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=5)
        ax.axis("off")

    #Ligne 4 : vecteurs de mouvement + residu
    p_idx = next((i for i, r in enumerate(records) if r["type"] == "P"), None)
    ax4a = fig.add_subplot(5, n_show, 3*n_show + 1)
    ax4a.set_ylabel("4 & 5] motion + residual", rotation=0, labelpad=55, fontsize=8, va="center")

    if p_idx is not None:
        # Image de fond = frame originale
        ax4a = fig.add_subplot(5, n_show, 3*n_show + 1, projection=None)
        ax4a.imshow(_rgb(orig_frames[p_idx]))
        mv  = records[p_idx]["mv"]
        mb  = params.macroblock
        mbh, mbw = mv.shape[:2]
        xs  = np.arange(mbw) * mb + mb // 2
        ys  = np.arange(mbh) * mb + mb // 2
        X, Y_g = np.meshgrid(xs, ys)
        DX = mv[..., 1].astype(float)
        DY = mv[..., 0].astype(float)
        ax4a.quiver(X, Y_g, DX, DY, color="red", scale=200,
                    width=0.004, headwidth=4)
        ax4a.set_title(f"motion vectors on P-frame {p_idx}", fontsize=8)
        ax4a.axis("off")

        #Carte des residus
        ax4b = fig.add_subplot(5, n_show, 3*n_show + n_show)
        prev_gray = cv2.cvtColor(orig_frames[p_idx - 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        curr_gray = cv2.cvtColor(orig_frames[p_idx],     cv2.COLOR_BGR2GRAY).astype(np.float32)
        res = curr_gray - prev_gray
        im2 = ax4b.imshow(res, cmap="RdBu", vmin=-40, vmax=40)
        ax4b.set_title(f"residual (P-frame {p_idx} vs prev)", fontsize=8)
        plt.colorbar(im2, ax=ax4b, fraction=0.046).ax.tick_params(labelsize=5)
        ax4b.axis("off")

    #Ligne 5 : frames reconstruites
    for i in range(n_show):
        ax = fig.add_subplot(5, n_show, 4*n_show + i + 1)
        ax.imshow(_rgb(recon_frames[i]))
        ftype = records[i]["type"] if i < len(records) else "?"
        p_val = psnr(orig_frames[i], recon_frames[i])
        ax.set_title(f"rec {i} [{ftype}]\n{p_val:.1f}dB", fontsize=7)
        if i == 0:
            ax.set_ylabel("5] reconstructions", rotation=0, labelpad=55, fontsize=8, va="center")
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def make_sweep_figure(frames, out_path):
    """
    Deux graphes :
      - Ratio de compression vs facteur de qualite
      - Ratio de compression vs taille du GOP
    """
    orig_bytes = sum(f.nbytes for f in frames)

    # sweep qualité (Q de 10 a 90)
    q_vals   = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    q_ratios = []
    q_psnrs  = []
    for q in q_vals:
        p = Params(gop=8, quality=q, block=8, macroblock=16, search=8, subsample=True)
        blob = encode(frames, p)
        recon, _, records = decode(blob, frames[0].shape[:2])
        q_ratios.append(orig_bytes / len(blob))
        q_psnrs.append(np.mean([psnr(o, r) for o, r in zip(frames, recon)]))
        print(f"  Q={q:3d}  ratio={q_ratios[-1]:.1f}x  PSNR={q_psnrs[-1]:.1f}dB")

    # sweep GOP (1 a 16)
    gop_vals   = [1, 2, 4, 6, 8, 10, 12, 16]
    gop_ratios = []
    gop_psnrs  = []
    for g in gop_vals:
        p = Params(gop=g, quality=50, block=8, macroblock=16, search=8, subsample=True)
        blob = encode(frames, p)
        recon, _, records = decode(blob, frames[0].shape[:2])
        gop_ratios.append(orig_bytes / len(blob))
        gop_psnrs.append(np.mean([psnr(o, r) for o, r in zip(frames, recon)]))
        print(f"  GOP={g:2d}  ratio={gop_ratios[-1]:.1f}x  PSNR={gop_psnrs[-1]:.1f}dB")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Experimental sweeps", fontsize=12, fontweight="bold")

    axes[0].plot(q_vals, q_ratios, "o-b", linewidth=2, markersize=6)
    axes[0].set_xlabel("quality", fontsize=10)
    axes[0].set_ylabel("compression ratio (x)", fontsize=10)
    axes[0].set_title(f"ratio vs quality (GOP=8)", fontsize=10)
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(gop_vals, gop_ratios, "s-r", linewidth=2, markersize=6)
    axes[1].set_xlabel("GOP size", fontsize=10)
    axes[1].set_ylabel("compression ratio (x)", fontsize=10)
    axes[1].set_title(f"ratio vs GOP (Q=50)", fontsize=10)
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
