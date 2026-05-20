import argparse
import os
import sys
import cv2
import numpy as np

from mpeg_codec import (encode, decode, psnr, frame_breakdown,
                        Params, DEFAULT_PARAMS)


def load_frames(folder):
    """Charge toutes les images d'un dossier, triees par nom."""
    exts = ('.png', '.jpg', '.jpeg', '.bmp')
    files = sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in exts
    )
    if not files:
        sys.exit(f"[ERREUR] Aucune image trouvee dans {folder}")
    frames = []
    for f in files:
        img = cv2.imread(os.path.join(folder, f))
        if img is not None:
            frames.append(img)
    print(f"  {len(frames)} frames chargees depuis '{folder}'")
    return frames


def cmd_encode(args):
    frames = load_frames(args.frames_dir)
    params = Params(
        gop=args.gop, quality=args.q,
        block=8, macroblock=16, search=8, subsample=True
    )
    print(f"  Encodage : GOP={params.gop}  qualite={params.quality}")
    blob = encode(frames, params)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(blob)
    orig_bytes = sum(fr.nbytes for fr in frames)
    ratio = orig_bytes / len(blob)
    print(f"  Taille originale  : {orig_bytes:,} octets")
    print(f"  Taille compressee : {len(blob):,} octets")
    print(f"  Ratio             : {ratio:.1f}x")
    print(f"  Bitstream ecrit   : {args.output}")


def cmd_decode(args):
    with open(args.input, "rb") as f:
        blob = f.read()
    ref_frames = load_frames(args.ref) if args.ref else None
    out_shape = ref_frames[0].shape[:2] if ref_frames else None
    print("  Decodage en cours...")
    recon_frames, params, records = decode(blob, out_shape)
    n_i, n_p = frame_breakdown(records)
    print(f"  I-frames : {n_i}   P-frames : {n_p}")
    if ref_frames:
        psnrs = [psnr(o, r) for o, r in zip(ref_frames, recon_frames)]
        print(f"  PSNR moyen : {np.mean(psnrs):.2f} dB")
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        for i, fr in enumerate(recon_frames):
            cv2.imwrite(os.path.join(args.output, f"dec_{i:04d}.png"), fr)
        print(f"  Frames reconstruites -> {args.output}/")


def cmd_viz(args):
    from viz import make_pipeline_figure
    frames = load_frames(args.frames_dir)
    with open(args.bitstream, "rb") as f:
        blob = f.read()
    out_shape = frames[0].shape[:2]
    recon, params, records = decode(blob, out_shape)
    make_pipeline_figure(frames, recon, records, params, args.output)
    print(f"  Figure sauvegardee : {args.output}")


def cmd_sweep(args):
    from viz import make_sweep_figure
    frames = load_frames(args.frames_dir)
    make_sweep_figure(frames, args.output)
    print(f"  Figure d'analyse sauvegardee : {args.output}")


def main():
    p = argparse.ArgumentParser(description="MPEG-4-lite — M1 IL USTHB")
    sub = p.add_subparsers(dest="cmd")

    pe = sub.add_parser("encode")
    pe.add_argument("frames_dir")
    pe.add_argument("-o", "--output", default="video.bin")
    pe.add_argument("--gop",  type=int, default=8)
    pe.add_argument("--q",    type=int, default=50)

    pd = sub.add_parser("decode")
    pd.add_argument("input")
    pd.add_argument("-o", "--output", default=None)
    pd.add_argument("--ref", default=None)

    pv = sub.add_parser("viz")
    pv.add_argument("frames_dir")
    pv.add_argument("bitstream")
    pv.add_argument("-o", "--output", default="pipeline.png")

    ps = sub.add_parser("sweep")
    ps.add_argument("frames_dir")
    ps.add_argument("-o", "--output", default="experiments.png")

    args = p.parse_args()
    if args.cmd == "encode":  cmd_encode(args)
    elif args.cmd == "decode": cmd_decode(args)
    elif args.cmd == "viz":    cmd_viz(args)
    elif args.cmd == "sweep":  cmd_sweep(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
