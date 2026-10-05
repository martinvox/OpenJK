# HD texture pipeline for Jedi Academy

Extracts the game's textures, AI-upscales them with
[Real-ESRGAN ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN) (portable exe,
runs on any Vulkan GPU — NVIDIA/AMD/Intel, no CUDA/Python ML stack needed) and
packs them into an override pk3, `GameData/base/zz_hd_textures.pk3`, which the
engine loads after `assets*.pk3`.

Only the scripts live in this repo. The game textures are copyrighted and are
never committed; everything generated stays in your local `work_dir`.

## Quick start

Requirements: Windows, Python 3.10+ and a Vulkan-capable GPU.

```bat
python -m pip install --user pillow numpy
tools\hd-textures\hdtex.bat
```

`hdtex.bat` runs `hdtex.py all`, which does the following:

1. **setup**: downloads Real-ESRGAN ncnn-vulkan (v0.2.5.0) and the extra models
   into `work_dir/bin` if they are missing.
2. **extract**: copies every `.jpg/.tga/.png` out of `assets*.pk3` into
   `work_dir/src`, keeping the exact paths. The pk3s are only opened for
   reading. When a file exists in several pk3s, the alphabetically last pk3 wins,
   which matches the engine's own resolution order.
3. **plan**: decides per texture whether to upscale it and by how much, and
   writes `plan.json` and `skipped.txt`.
4. **upscale**: runs the GPU in batches and saves results to `work_dir/out`.
5. **pack**: writes `GameData/base/zz_hd_textures.pk3` atomically.

**Everything is resumable.** If you interrupt it, run the same command again
and it continues where it stopped. Each finished texture is recorded in
`state.json` together with a hash of its source CRC and the settings that
affect it. If you change the model, padding, alpha or JPG settings, or the game
data changes, only the affected textures are redone.

Other commands:

| Command | What it does |
|---|---|
| `python hdtex.py compare` | Upscales the textures listed in `samples.txt` with several models and writes side-by-side sheets to `work_dir/compare/sheets/`. Use `--models a,b,c`, `--cell 384`, `--corner`, `--force`. |
| `python hdtex.py upscale --limit 50` | Processes only the next 50 textures (for testing). |
| `python hdtex.py pack` | Re-packs whatever is finished. |
| `python hdtex.py unpack` | Removes the generated pk3(s) from `base/` and restores the vanilla look. The originals are never touched. |

## In-game settings

OpenJK reads its config from `Documents/My Games/OpenJK/base/openjk_sp.cfg` (SP)
or `openjk.cfg` (MP), not the `jaconfig.cfg`/`jampconfig.cfg` in `GameData/base`.
OpenJK's defaults are already fine, but check these:

- `r_picmip 0`: `1` halves every texture, which throws away most of the gain.
- `r_ext_compress_textures 0`: uncompressed gives the best quality and needs
  more VRAM. Set it to `1` (DXT) if VRAM runs out.
- `r_texturebits 32`, `r_textureMode GL_LINEAR_MIPMAP_LINEAR`,
  `r_ext_texture_filter_anisotropic 16`.

## Multiplayer

Servers running with `sv_pure 1` refuse clients that load pk3 files the server
doesn't have. The HD pack is meant for single player and for your own or local
servers. To use it on a pure server, the server must have the same pk3.

## How textures are processed

**Scale.** Each texture gets the largest factor from `scales` (`[4, 2]`) that
keeps its longest side ≤ `max_side` (2048). Textures with a side ≥
`skip_if_max_side_ge` (1024) are left alone. Every original JA texture has
power-of-two dimensions, so 2× and 4× outputs stay powers of two.

The AI models are native 4×. For 2× the 4× result is Lanczos-downscaled.

**Skips**, all configurable in `config.json`:

- `skip_patterns`: glob patterns on the in-pk3 path. The defaults are `fonts/*`
  (bitmap fonts must keep their glyph grid), `video/*`, `*lightmap*`, `maps/*`,
  `gfx/2d/charsgrid*` (console font) and `gfx/colors/*`.
  `force_include_patterns` overrides them.
- `min_side` / `min_pixels`: skips tiny images and 1–4 px gradient strips.
- `skip_flat_stddev`: skips single-colour images, measured on RGB *and* alpha.

`work_dir/skipped.txt` lists every skipped file and the reason.

**Seamless tiling.** Before upscaling, each image is padded by `pad_px` pixels
and the padding is cropped off afterwards. World textures (`wrap_pad_patterns`,
default `textures/*`) are padded by wrapping around, so tiling textures stay
seamless. Everything else (model skins, UI, effects) and skies
(`edge_pad_patterns`) use edge padding.

**Alpha** (`.tga`/`.png` with transparency):

- The RGB goes through the AI model. The alpha is scaled separately with a
  non-hallucinating filter (`alpha_filter`: `lanczos` by default, clamped to 0–255).
- `alpha_bleed` (on by default) prevents dark halos. Before upscaling, the
  colour of fully transparent pixels is replaced with the colour of the nearest
  visible pixels, so the AI doesn't pull black or garbage colours into the
  visible edge. Where the final alpha is still 0, the original colour is put
  back, so blend modes that read RGB under alpha 0 behave as before.

**Output** keeps the same path, file name and extension:

- JPG: quality `jpg_quality` (95), chroma subsampling `jpg_subsampling` (4:4:4).
- TGA: uncompressed, 32-bit when it has alpha, 24-bit otherwise.
- PNG: RGBA or RGB as in the original.

## Models

The default model is set by `"model"`. `model_overrides` assigns other models
by pattern, for example:

```json
"model_overrides": [ { "pattern": "gfx/*", "model": "RealESRGAN_General_WDN_x4_v3" } ]
```

Available out of the box:

| Model | Character |
|---|---|
| `4x_NMKD-Siax_200k` | Trained for realistic textures. Adds plausible surface detail and skin texture. Can add faint speckles on very dark, smooth metal. |
| `RealESRGAN_General_WDN_x4_v3` | Clean and conservative with sharp edges. Denoises JPEG artifacts and invents little. |
| `realesrgan-x4plus` | Photo model. Gives painterly, waxy results on textures and hallucinates rays on flares. Not recommended. |

Any ncnn `.param`/`.bin` pair dropped into `work_dir/bin/realesrgan/models` can
be used by name, for example models from
[upscayl/custom-models](https://github.com/upscayl/custom-models).

## Pack size and splitting

A full run at `max_side` 2048 produces roughly 4–5 GB. OpenJK reads pk3s with
32-bit minizip, so the packer splits the output into parts of at most
`pk3_split_mb` (1800 MB):

- `zz_hd_textures.pk3`
- `zz_hd_textures_02.pk3`
- …

All parts sort after `assets*.pk3`. Stale extra parts from an earlier, larger
run are removed. TGAs are deflated; JPG/PNG are stored, since they are already
compressed.

## Configuration reference (`config.json`)

Personal overrides can go in `config.local.json` next to it, which is
git-ignored and merged on top of `config.json`.

| Key | Meaning |
|---|---|
| `game_base` | `GameData/base` folder containing `assets*.pk3` |
| `work_dir` | Where `bin/`, `src/`, `out/`, `compare/` and the state files live. **Must not be inside `base/`**, because the engine loads loose files there. |
| `pk3_glob`, `extensions` | Which pk3s and file types to read |
| `model`, `model_overrides` | Upscaler model(s) |
| `gpu_id` | `"auto"` or a Vulkan device index |
| `tile_size` | `0` = auto. Set to e.g. `256` if you get out-of-memory errors. |
| `batch_size`, `workers` | Images per GPU call; CPU processes for pre- and post-processing (`0` = cores − 2) |
| `scales`, `max_side`, `skip_if_max_side_ge` | Scaling policy |
| `min_side`, `min_pixels`, `skip_flat_stddev`, `skip_patterns`, `force_include_patterns` | Skip rules |
| `wrap_pad_patterns`, `edge_pad_patterns`, `pad_px` | Seam handling |
| `alpha_filter`, `alpha_bleed` | Alpha handling |
| `jpg_quality`, `jpg_subsampling`, `png_compress_level` | Output encoding |
| `pk3_name`, `pk3_split_mb` | Output pk3 |
