#!/usr/bin/env python3
"""
Jedi Academy HD texture pipeline.

Extracts textures from the original assets*.pk3 (read-only), AI-upscales them
with Real-ESRGAN ncnn-vulkan and packs the results into an override pk3
(zz_hd_textures.pk3) that the engine loads after the original assets.

Every stage is resumable. Run `hdtex.py all` to (re)run the whole pipeline.
See README.md for details.
"""
import argparse
import concurrent.futures as cf
import fnmatch
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

REALESRGAN_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
                  "realesrgan-ncnn-vulkan-20220424-windows.zip")
# Extra ncnn models (param+bin) that are not bundled with Real-ESRGAN.
EXTRA_MODELS_URL = "https://github.com/upscayl/custom-models/raw/main/models/"
EXTRA_MODELS = ["4x_NMKD-Siax_200k", "RealESRGAN_General_WDN_x4_v3"]

# Bump when the processing code changes in a way that should invalidate outputs.
PIPELINE_VERSION = 1
INFO_VERSION = 2


# --------------------------------------------------------------------------- config

def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    local = os.path.splitext(path)[0] + ".local.json"
    if os.path.exists(local):
        with open(local, encoding="utf-8") as f:
            cfg.update(json.load(f))
    base = os.path.dirname(os.path.abspath(path))
    for k in ("game_base", "work_dir"):
        cfg[k] = os.path.normpath(os.path.join(base, cfg[k]))
    w = cfg["work_dir"]
    cfg["bin_dir"] = os.path.join(w, "bin")
    cfg["src_dir"] = os.path.join(w, "src")
    cfg["out_dir"] = os.path.join(w, "out")
    cfg["stage_dir"] = os.path.join(w, "stage")
    cfg["exe"] = os.path.join(cfg["bin_dir"], "realesrgan", "realesrgan-ncnn-vulkan.exe")
    cfg["models_dir"] = os.path.join(cfg["bin_dir"], "realesrgan", "models")
    # Safety: never let the work dir live inside base/ — loose files there are loaded by the game.
    if os.path.normcase(w).startswith(os.path.normcase(cfg["game_base"]) + os.sep):
        sys.exit("work_dir must not be inside game_base (the engine would load the loose files).")
    return cfg


def match_any(path, patterns):
    p = path.lower()
    return any(fnmatch.fnmatchcase(p, pat.lower()) for pat in patterns)


def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def load_json(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


# --------------------------------------------------------------------------- setup

def cmd_setup(cfg, args):
    os.makedirs(cfg["bin_dir"], exist_ok=True)
    if not os.path.exists(cfg["exe"]):
        print("Downloading Real-ESRGAN ncnn-vulkan ...")
        data = urllib.request.urlopen(REALESRGAN_URL).read()
        zipfile.ZipFile(io.BytesIO(data)).extractall(os.path.join(cfg["bin_dir"], "realesrgan"))
    for m in EXTRA_MODELS:
        for ext in ("param", "bin"):
            dst = os.path.join(cfg["models_dir"], f"{m}.{ext}")
            if not os.path.exists(dst):
                print(f"Downloading model {m}.{ext} ...")
                urllib.request.urlretrieve(EXTRA_MODELS_URL + f"{m}.{ext}", dst + ".part")
                os.replace(dst + ".part", dst)
    print("Real-ESRGAN:", cfg["exe"])
    print("Models:", sorted({os.path.splitext(f)[0] for f in os.listdir(cfg["models_dir"])}))


# --------------------------------------------------------------------------- extract

def cmd_extract(cfg, args):
    """Copy every texture out of the pk3s (later pk3 alphabetically wins), resumable."""
    base = cfg["game_base"]
    paks = sorted((f for f in os.listdir(base) if fnmatch.fnmatch(f.lower(), cfg["pk3_glob"].lower())),
                  key=str.lower)
    exts = tuple(e.lower() for e in cfg["extensions"])
    winners = {}
    for pak in paks:
        with zipfile.ZipFile(os.path.join(base, pak)) as z:
            for info in z.infolist():
                if not info.is_dir() and info.filename.lower().endswith(exts):
                    winners[info.filename.lower()] = (pak, info.filename, info.CRC)
    manifest_path = os.path.join(cfg["work_dir"], "manifest.json")
    old = load_json(manifest_path, {})
    manifest, extracted = {}, 0
    by_pak = {}
    for key, (pak, name, crc) in winners.items():
        by_pak.setdefault(pak, []).append((key, name, crc))
    for pak, items in by_pak.items():
        with zipfile.ZipFile(os.path.join(base, pak)) as z:  # opened read-only
            for key, name, crc in items:
                dst = os.path.join(cfg["src_dir"], name)
                prev = old.get(key)
                if not (prev and prev["crc"] == crc and os.path.exists(dst)):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    with open(dst + ".part", "wb") as f:
                        f.write(z.read(name))
                    os.replace(dst + ".part", dst)
                    extracted += 1
                manifest[key] = {"pak": pak, "name": name, "crc": crc}
    save_json(manifest_path, manifest)
    print(f"{len(manifest)} unique textures from {len(paks)} pk3s ({extracted} newly extracted)")


# --------------------------------------------------------------------------- plan

def image_info(path):
    with Image.open(path) as im:
        has_alpha = "A" in im.getbands() or (im.mode == "P" and "transparency" in im.info)
        px = np.asarray(im.convert("RGBA" if has_alpha else "RGB"), dtype=np.float32)
        return im.size, im.mode, has_alpha, float(px.std(axis=(0, 1)).max())  # alpha counts too


def decide(cfg, name, size, std):
    """Return (scale, reason). scale == 0 means skip."""
    w, h = size
    if match_any(name, cfg["skip_patterns"]) and not match_any(name, cfg["force_include_patterns"]):
        return 0, "skip_pattern"
    if max(w, h) >= cfg["skip_if_max_side_ge"]:
        return 0, "already_large"
    if min(w, h) < cfg["min_side"] or w * h < cfg["min_pixels"]:
        return 0, "too_small"
    if std < cfg["skip_flat_stddev"]:
        return 0, "flat_color"
    for s in sorted(cfg["scales"], reverse=True):
        if max(w, h) * s <= cfg["max_side"]:
            return s, "upscale"
    return 0, "exceeds_max_side"


def model_for(cfg, name):
    for o in cfg["model_overrides"]:
        if match_any(name, [o["pattern"]]):
            return o["model"]
    return cfg["model"]


def _info_job(args):
    key, path = args
    try:
        return key, image_info(path), None
    except Exception as e:  # unreadable/corrupt image
        return key, None, str(e)


def cmd_plan(cfg, args, quiet=False):
    manifest = load_json(os.path.join(cfg["work_dir"], "manifest.json"), None)
    if manifest is None:
        sys.exit("Run `extract` first.")
    cache_path = os.path.join(cfg["work_dir"], "info_cache.json")
    cache = load_json(cache_path, {})
    todo = [(k, os.path.join(cfg["src_dir"], m["name"])) for k, m in manifest.items()
            if cache.get(k, {}).get("crc") != m["crc"] or cache[k].get("v") != INFO_VERSION]
    if todo:
        with cf.ProcessPoolExecutor(workers(cfg)) as ex:
            for key, info, err in ex.map(_info_job, todo, chunksize=32):
                cache[key] = {"crc": manifest[key]["crc"], "v": INFO_VERSION, "info": info, "err": err}
        save_json(cache_path, cache)
    plan, counts = {}, {}
    for key, m in sorted(manifest.items()):
        c = cache[key]
        if c["err"]:
            scale, reason, size, mode, alpha = 0, "unreadable", None, None, False
        else:
            size, mode, alpha, std = c["info"]
            scale, reason = decide(cfg, m["name"], size, std)
        plan[key] = {"name": m["name"], "crc": m["crc"], "size": size, "mode": mode,
                     "alpha": alpha, "scale": scale, "reason": reason, "model": model_for(cfg, m["name"])}
        counts[reason] = counts.get(reason, 0) + 1
    save_json(os.path.join(cfg["work_dir"], "plan.json"), plan)
    with open(os.path.join(cfg["work_dir"], "skipped.txt"), "w", encoding="utf-8") as f:
        for p in plan.values():
            if not p["scale"]:
                f.write(f"{p['reason']:16} {p['size']}  {p['name']}\n")
    if not quiet:
        print("Plan:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        print("Skipped files are listed in", os.path.join(cfg["work_dir"], "skipped.txt"))
    return plan


# --------------------------------------------------------------------------- image processing

def bleed_rgb(rgb, alpha, iterations=64):
    """Fill RGB of fully transparent pixels with the colour of nearby visible pixels,
    so the upscaler does not drag dark/garbage colours into the visible edge (halos)."""
    known = alpha > 0
    if known.all() or not known.any():
        return rgb
    rgb = rgb.astype(np.float32).copy()
    filled = known.copy()
    for _ in range(iterations):
        if filled.all():
            break
        acc = np.zeros_like(rgb)
        cnt = np.zeros(filled.shape, np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                m = np.roll(np.roll(filled, dy, 0), dx, 1)
                acc += np.roll(np.roll(rgb, dy, 0), dx, 1) * m[..., None]
                cnt += m
        grow = (~filled) & (cnt > 0)
        rgb[grow] = acc[grow] / cnt[grow][:, None]
        filled |= grow
    if not filled.all():
        rgb[~filled] = rgb[filled].mean(axis=0)
    return rgb.clip(0, 255).astype(np.uint8)


def load_rgba(path):
    im = Image.open(path)
    im.load()
    if im.mode == "P":
        im = im.convert("RGBA" if "transparency" in im.info else "RGB")
    if im.mode in ("LA", "PA", "RGBa"):
        im = im.convert("RGBA")
    elif im.mode not in ("RGB", "RGBA", "L"):
        im = im.convert("RGB")
    return im


def pad_mode(cfg, name):
    if match_any(name, cfg["edge_pad_patterns"]):
        return "edge"
    return "wrap" if match_any(name, cfg["wrap_pad_patterns"]) else "edge"


def prepare(job):
    """Write the padded RGB image that the GPU upscaler will process."""
    cfg, name, src, stage_png = job["cfg"], job["name"], job["src"], job["stage"]
    im = load_rgba(src)
    rgb = np.asarray(im.convert("RGB"))
    if im.mode == "RGBA" and cfg["alpha_bleed"]:
        rgb = bleed_rgb(rgb, np.asarray(im.getchannel("A")))
    p = cfg["pad_px"]
    rgb = np.pad(rgb, ((p, p), (p, p), (0, 0)), mode=pad_mode(cfg, name))
    Image.fromarray(rgb).save(stage_png, compress_level=1)


def finish(job):
    """Crop the upscaled RGB, recombine with a Lanczos-scaled alpha and save in the original format."""
    cfg, src, up_png, dst, scale = job["cfg"], job["src"], job["up"], job["dst"], job["scale"]
    im = load_rgba(src)
    w, h = im.size
    tw, th = w * scale, h * scale
    up = Image.open(up_png).convert("RGB")
    f = up.width // (w + 2 * cfg["pad_px"])  # model's native factor (4)
    p = cfg["pad_px"] * f
    up = up.crop((p, p, p + w * f, p + h * f))
    if up.size != (tw, th):
        up = up.resize((tw, th), Image.LANCZOS)
    if im.mode == "RGBA":
        resample = {"lanczos": Image.LANCZOS, "bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR}[cfg["alpha_filter"]]
        a = im.getchannel("A").resize((tw, th), resample)  # PIL clamps to 0..255, no overshoot
        if cfg["alpha_bleed"]:
            # Where the result is fully transparent keep the original (non-bled) colour,
            # so additive/odd blend modes that read RGB under alpha 0 still behave.
            orig = np.asarray(im.convert("RGB").resize((tw, th), Image.LANCZOS))
            rgb = np.asarray(up).copy()
            zero = np.asarray(a) == 0
            rgb[zero] = orig[zero]
            up = Image.fromarray(rgb)
        out = up.convert("RGBA")
        out.putalpha(a)
    elif im.mode == "L":
        out = up.convert("L")
    else:
        out = up
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    ext = os.path.splitext(dst)[1].lower()
    if ext == ".jpg":
        out.convert("L" if out.mode == "L" else "RGB").save(
            tmp, "JPEG", quality=cfg["jpg_quality"], subsampling=cfg["jpg_subsampling"], optimize=True)
    elif ext == ".tga":
        out.save(tmp, "TGA", rle=False)  # uncompressed, 32-bit when it has alpha
    else:
        out.save(tmp, "PNG", compress_level=cfg["png_compress_level"])
    os.replace(tmp, dst)


def workers(cfg):
    return cfg["workers"] or max(1, (os.cpu_count() or 4) - 2)


def run_realesrgan(cfg, model, in_dir, out_dir):
    if not os.path.exists(cfg["exe"]):
        sys.exit("Real-ESRGAN not found. Run `hdtex.py setup` first.")
    cmd = [cfg["exe"], "-i", in_dir, "-o", out_dir, "-n", model, "-s", "4", "-f", "png",
           "-m", cfg["models_dir"], "-t", str(cfg["tile_size"])]
    if str(cfg["gpu_id"]) != "auto":
        cmd += ["-g", str(cfg["gpu_id"])]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise RuntimeError(f"realesrgan-ncnn-vulkan failed with code {r.returncode}")


def settings_key(cfg, item):
    keys = ["pad_px", "alpha_filter", "alpha_bleed", "jpg_quality", "jpg_subsampling",
            "wrap_pad_patterns", "edge_pad_patterns"]
    blob = json.dumps([PIPELINE_VERSION, item["crc"], item["scale"], item["model"],
                       [cfg[k] for k in keys]], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def upscale_items(cfg, items, out_root, state=None, state_path=None, label=""):
    """items: list of plan entries. Runs in batches; results land in out_root/<name>."""
    stage_in = os.path.join(cfg["stage_dir"], "in")
    stage_out = os.path.join(cfg["stage_dir"], "out")
    by_model = {}
    for it in items:
        by_model.setdefault(it["model"], []).append(it)
    total, done = len(items), 0
    t0 = time.time()
    with cf.ProcessPoolExecutor(workers(cfg)) as pool:
        for model, its in by_model.items():
            for b in range(0, len(its), cfg["batch_size"]):
                batch = its[b:b + cfg["batch_size"]]
                for d in (stage_in, stage_out):
                    shutil.rmtree(d, ignore_errors=True)
                    os.makedirs(d)
                jobs = []
                for i, it in enumerate(batch):
                    jobs.append({"cfg": cfg, "name": it["name"], "scale": it["scale"],
                                 "src": os.path.join(cfg["src_dir"], it["name"]),
                                 "stage": os.path.join(stage_in, f"{i:05d}.png"),
                                 "up": os.path.join(stage_out, f"{i:05d}.png"),
                                 "dst": os.path.join(out_root, it["name"])})
                list(pool.map(prepare, jobs))
                run_realesrgan(cfg, model, stage_in, stage_out)
                for job, err in zip(jobs, pool.map(_safe_finish, jobs)):
                    if err:
                        print(f"  ! {job['name']}: {err}")
                done += len(batch)
                if state is not None:
                    for it, job in zip(batch, jobs):
                        if os.path.exists(job["dst"]):
                            state[it["name"].lower()] = settings_key(cfg, it)
                    save_json(state_path, state)
                el = time.time() - t0
                eta = el / done * (total - done)
                print(f"{label}[{model}] {done}/{total}  {el/60:.1f} min elapsed, ~{eta/60:.1f} min left",
                      flush=True)
    shutil.rmtree(cfg["stage_dir"], ignore_errors=True)


def _safe_finish(job):
    try:
        finish(job)
        return None
    except Exception as e:
        return str(e)


def cmd_upscale(cfg, args):
    plan = cmd_plan(cfg, args)
    state_path = os.path.join(cfg["work_dir"], "state.json")
    state = load_json(state_path, {})
    todo = [p for k, p in plan.items()
            if p["scale"] and (state.get(k) != settings_key(cfg, p)
                               or not os.path.exists(os.path.join(cfg["out_dir"], p["name"])))]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} textures to upscale ({sum(1 for p in plan.values() if p['scale'])} planned in total)")
    if todo:
        upscale_items(cfg, todo, cfg["out_dir"], state, state_path)


# --------------------------------------------------------------------------- pack

def cmd_pack(cfg, args):
    plan = load_json(os.path.join(cfg["work_dir"], "plan.json"), None)
    if plan is None:
        sys.exit("Nothing planned yet; run `upscale` first.")
    state = load_json(os.path.join(cfg["work_dir"], "state.json"), {})
    files = []
    for k, p in sorted(plan.items()):
        path = os.path.join(cfg["out_dir"], p["name"])
        if p["scale"] and state.get(k) == settings_key(cfg, p) and os.path.exists(path):
            files.append((p["name"], path))
    if not files:
        sys.exit("No upscaled textures to pack.")
    base_name = os.path.splitext(cfg["pk3_name"])[0]
    limit = cfg["pk3_split_mb"] * 1024 * 1024
    groups, cur, size = [], [], 0
    for name, path in files:
        s = os.path.getsize(path)
        if cur and size + s > limit:
            groups.append(cur)
            cur, size = [], 0
        cur.append((name, path))
        size += s
    groups.append(cur)
    targets = [cfg["pk3_name"]] + [f"{base_name}_{i:02d}.pk3" for i in range(2, len(groups) + 1)]
    for tgt, group in zip(targets, groups):
        dst = os.path.join(cfg["game_base"], tgt)
        if os.path.basename(dst).lower().startswith("assets"):
            sys.exit("Refusing to write over an original assets pk3.")
        tmp = dst + ".tmp"
        with zipfile.ZipFile(tmp, "w") as z:
            for name, path in group:
                ct = zipfile.ZIP_DEFLATED if name.lower().endswith(".tga") else zipfile.ZIP_STORED
                z.write(path, name.replace("\\", "/"), compress_type=ct)
        os.replace(tmp, dst)
        print(f"Wrote {dst}: {len(group)} files, {os.path.getsize(dst)/2**20:.0f} MB")
    # Remove stale split parts from a previous, larger run.
    i = len(groups) + 1
    while os.path.exists(os.path.join(cfg["game_base"], f"{base_name}_{i:02d}.pk3")):
        os.remove(os.path.join(cfg["game_base"], f"{base_name}_{i:02d}.pk3"))
        print(f"Removed stale {base_name}_{i:02d}.pk3")
        i += 1


def cmd_unpack(cfg, args):
    """Remove the generated pk3(s) from the game folder (the originals are never touched)."""
    base_name = os.path.splitext(cfg["pk3_name"])[0].lower()
    for f in os.listdir(cfg["game_base"]):
        fl = f.lower()
        if fl == cfg["pk3_name"].lower() or (fl.startswith(base_name + "_") and fl.endswith(".pk3")):
            os.remove(os.path.join(cfg["game_base"], f))
            print("Removed", f)


# --------------------------------------------------------------------------- compare

def checker(size, cell=16):
    w, h = size
    y, x = np.mgrid[0:h, 0:w]
    c = (((x // cell) + (y // cell)) % 2) * 60 + 100
    return Image.fromarray(np.stack([c] * 3, -1).astype(np.uint8))


def show(im, size, bg):
    im = im.convert("RGBA")
    if bg == "checker":
        base = checker(size).convert("RGBA")
    else:
        base = Image.new("RGBA", size, bg)
    return Image.alpha_composite(base, im.resize(size, Image.NEAREST) if im.size != size else im).convert("RGB")


def cmd_compare(cfg, args):
    from PIL import ImageDraw, ImageFont
    plan = cmd_plan(cfg, args, quiet=True)
    with open(args.list, encoding="utf-8") as f:
        names = [l.strip().lower() for l in f if l.strip() and not l.startswith("#")]
    models = args.models.split(",")
    root = os.path.join(cfg["work_dir"], "compare")
    items = []
    for n in names:
        p = dict(plan[n])
        if not p["scale"]:  # samples are compared even if the plan would skip them
            p["scale"] = 4 if max(p["size"]) * 4 <= cfg["max_side"] else 2
            print(f"note: {n} would be skipped ({plan[n]['reason']}); comparing anyway")
        items.append(p)
    for m in models:
        need = [dict(p, model=m) for p in items
                if not os.path.exists(os.path.join(root, m, p["name"])) or args.force]
        if need:
            upscale_items(cfg, need, os.path.join(root, m), label="compare ")
    # Build comparison sheets: one row per texture, crops at output resolution.
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        small = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = small = ImageFont.load_default()
    cell, gap, head = args.cell, 6, 26
    cols = ["original (nearest)", "lanczos"] + models
    sheets_dir = os.path.join(root, "sheets")
    os.makedirs(sheets_dir, exist_ok=True)
    per = args.per_sheet
    for s in range(0, len(items), per):
        rows = items[s:s + per]
        W = len(cols) * (cell + gap) + gap
        H = head + sum((cell * (2 if r["alpha"] else 1) + head + gap) for r in rows) + gap
        sheet = Image.new("RGB", (W, H), (24, 24, 28))
        d = ImageDraw.Draw(sheet)
        for i, c in enumerate(cols):
            d.text((gap + i * (cell + gap) + 4, 4), c, fill=(230, 230, 230), font=font)
        y = head
        for r in rows:
            src = load_rgba(os.path.join(cfg["src_dir"], r["name"]))
            w, h = src.size
            tw, th = w * r["scale"], h * r["scale"]
            outs = [src.resize((tw, th), Image.NEAREST), src.resize((tw, th), Image.LANCZOS)]
            outs += [load_rgba(os.path.join(root, m, r["name"])) for m in models]
            # Crop a region of `cell` px from the output (centre, or full image if small).
            cw, ch = min(cell, tw), min(cell, th)
            cx = (tw - cw) // 2 if not args.corner else 0
            cy = (th - ch) // 2 if not args.corner else 0
            d.text((gap, y + 3), f"{r['name']}   {w}x{h} -> {tw}x{th}   {r['mode']}"
                   f"{'  (alpha: top=checker, bottom=dark)' if r['alpha'] else ''}",
                   fill=(255, 210, 120), font=small)
            y += head
            for i, o in enumerate(outs):
                crop = o.crop((cx, cy, cx + cw, cy + ch))
                x = gap + i * (cell + gap)
                if r["alpha"]:
                    sheet.paste(show(crop, (cell, cell), "checker"), (x, y))
                    sheet.paste(show(crop, (cell, cell), (10, 10, 10, 255)), (x, y + cell))
                else:
                    sheet.paste(crop.convert("RGB").resize((cell, cell), Image.NEAREST), (x, y))
            y += cell * (2 if r["alpha"] else 1) + gap
        out = os.path.join(sheets_dir, f"compare_{s // per + 1:02d}.png")
        sheet.save(out)
        print("Wrote", out)


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup", help="download Real-ESRGAN ncnn-vulkan and extra models")
    sub.add_parser("extract", help="extract textures from the pk3s into work_dir/src")
    sub.add_parser("plan", help="decide what to upscale / skip (writes plan.json, skipped.txt)")
    u = sub.add_parser("upscale", help="upscale everything planned (resumable)")
    u.add_argument("--limit", type=int, default=0, help="only process N textures (testing)")
    sub.add_parser("pack", help="pack results into GameData/base/zz_hd_textures.pk3")
    sub.add_parser("unpack", help="remove the generated pk3(s) from the game folder")
    a = sub.add_parser("all", help="setup + extract + upscale + pack")
    a.add_argument("--limit", type=int, default=0)
    c = sub.add_parser("compare", help="upscale a sample list with several models and build comparison sheets")
    c.add_argument("--list", default=os.path.join(HERE, "samples.txt"))
    c.add_argument("--models", default="realesrgan-x4plus,4x_NMKD-Siax_200k,RealESRGAN_General_WDN_x4_v3")
    c.add_argument("--cell", type=int, default=384, help="crop size in output pixels")
    c.add_argument("--per-sheet", type=int, default=5)
    c.add_argument("--corner", action="store_true", help="crop the top-left corner instead of the centre")
    c.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.makedirs(cfg["work_dir"], exist_ok=True)
    if args.cmd == "all":
        cmd_setup(cfg, args)
        cmd_extract(cfg, args)
        cmd_upscale(cfg, args)
        cmd_pack(cfg, args)
    else:
        globals()["cmd_" + args.cmd](cfg, args)


if __name__ == "__main__":
    main()
