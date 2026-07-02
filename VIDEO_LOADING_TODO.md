# AutoDex Gallery Video Loading TODO

## Goal

Make the AutoDex gallery load quickly while preserving enough visual quality for browsing grasp episodes, camera views, and future interactive 3D reconstructions.

The gallery should default to lightweight preview media and only load the high-quality source media when explicitly needed.

## Current Bottlenecks

- The main gallery video currently points directly to Hugging Face MP4 files under `experiments/{hand}/{object}/{rank}/{camera}.mp4`.
- View thumbnails are currently implemented as `<video preload="metadata">`, so opening the view selector can trigger metadata requests for many camera videos.
- Pose thumbnails also use turntable videos, which means browsing poses can trigger additional video metadata loads.
- Large videos are acceptable for archival/download use, but they are too heavy as the default browsing layer.

## Recommended Direction

- Keep original-quality videos on Hugging Face as the archival source.
- Generate compressed preview MP4s for gallery playback.
- Generate still poster images for view and pose thumbnails.
- Make the page load only the selected main video by default.
- Use original videos only through an explicit high-quality option or fallback.

## TODO

### 1. Define Preview Media Profiles

- [ ] Define a default preview profile for the main video panel.
  - Candidate: H.264 MP4, 720p max width, 20 fps, CRF 30, no audio, `+faststart`.
- [ ] Define a lighter profile for a future side-by-side video + interactive 3D layout.
  - Candidate: H.264 MP4, 540p or 480p max width, 15-20 fps, CRF 31-33, no audio.
- [ ] Keep enough quality to inspect grasp contact and gross robot/object motion.
- [ ] Avoid replacing the original-quality MP4s.

### 2. Generate Preview Videos

- [ ] Add a script such as `scripts/build_gallery_previews.py`.
- [ ] Read `docs/experiments.json` and process every listed episode/camera.
- [ ] Skip preview files that already exist unless `--force` is passed.
- [ ] Write a JSON or CSV report with generated, skipped, and failed files.
- [ ] Use an ffmpeg command similar to:

```bash
ffmpeg -i input.mp4 -an \
  -vf "scale='min(1280,iw)':-2,fps=20" \
  -c:v libx264 -preset veryfast -crf 30 -pix_fmt yuv420p \
  -movflags +faststart output_preview.mp4
```

### 3. Generate Thumbnail Posters

- [ ] Generate still poster images for every view video.
  - Suggested path: `posters/experiments/{hand}/{object}/{rank}/{camera}.webp`
- [ ] Generate still poster images for every pose turntable.
  - Suggested path: `posters/turntable/{hand}/{object}/{rank}/turntable.webp`
- [ ] Use a representative timestamp rather than frame 0 if frame 0 is often uninformative.
- [ ] Prefer WebP for small file size, with JPG fallback if needed.

### 4. Upload Media Assets

- [ ] Upload preview MP4s to Hugging Face under a separate prefix, for example:
  - `preview/experiments/{hand}/{object}/{rank}/{camera}.mp4`
  - `preview/turntable/{hand}/{object}/{rank}/turntable.mp4`
- [ ] Upload poster images to Hugging Face or commit only small poster assets to GitHub if the total size stays reasonable.
- [ ] Do not commit large preview MP4 files to GitHub Pages.

### 5. Add Manifest and URL Helpers

- [ ] Add a media manifest, for example `docs/media_manifest.json`, or extend `experiments.json` with optional preview/poster availability.
- [ ] Add URL helpers in `docs/gallery.html`:
  - `previewVideoUrl(ep, camera)`
  - `sourceVideoUrl(ep, camera)`
  - `viewPosterUrl(ep, camera)`
  - `posePosterUrl(ep)`
- [ ] Fall back to the current original video URL when a preview is missing.

### 6. Replace Selector Thumbnail Videos

- [ ] Replace view selector `<video>` thumbnails with `<img>` poster thumbnails.
- [ ] Replace pose selector `<video>` thumbnails with poster images by default.
- [ ] Optionally play a low-quality preview only on hover/focus after a short delay.
- [ ] Keep selector rows pre-rendered, but avoid creating many active video elements.

### 7. Main Panel Loading Policy

- [ ] Load only the currently selected video in the main panel.
- [ ] Use compressed preview MP4 as the default source.
- [ ] Keep `preload="metadata"` or consider `preload="none"` if initial page load is still slow.
- [ ] Add an explicit high-quality option later if needed.
- [ ] For a side-by-side video + interactive 3D layout, use the lighter preview profile by default because the video panel is smaller.

### 8. Validation

- [ ] Compare page load behavior before and after with browser network tooling.
- [ ] Measure initial request count and transferred bytes.
- [ ] Verify that selector hover/open no longer triggers many MP4 metadata requests.
- [ ] Confirm that the selected main video still plays reliably.
- [ ] Check mobile layout and side-by-side desktop layout if/when the interactive 3D panel is shown together with video.

## Expected Impact

- Faster initial load.
- Faster selector expansion.
- Lower bandwidth usage.
- Less browser memory pressure from many thumbnail video elements.
- Original-quality media remains available for archival or detailed inspection.
