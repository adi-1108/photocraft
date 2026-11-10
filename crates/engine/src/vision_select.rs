//! Apple Vision for Select Subject, Object Selection's click, Remove Background and Mask All
//! Objects (Preferences › Performance › Use Apple Vision, shown only where Vision runs: macOS 14
//! and later). Vision's foreground objects come first; the built-in segmentation runs when the
//! preference is off, Vision isn't available (every other platform) or it finds nothing.

use std::sync::atomic::{AtomicBool, Ordering};

use photocraft_algo::segment::{Sampler, grabcut, subject};
use photocraft_algo::selection::Region;
use photocraft_geom::Rect;
use photocraft_raster::{Cancelled, Interrupt};

/// Longest side of the image Vision is given: it segments a small internal image anyway, and
/// the cap keeps the copy small on 30 MP documents. Masks are scaled back up to the document.
const WORK_SIDE: usize = 1536;

/// `performance.useAppleVision`, pushed by `Session::apply_prefs` (like the other engine-wide
/// budgets) so the Object Finder's hover lookups, which run without a session, follow it too.
static ENABLED: AtomicBool = AtomicBool::new(true);

pub(crate) fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed) && photocraft_vision::available()
}

/// Select Subject: every foreground object Vision finds in `area`, else the built-in subject
/// segmentation. `Err(Cancelled)` once `ctl` is cancelled.
pub(crate) fn subject_region(sampler: &dyn Sampler, area: Rect, ctl: &Interrupt) -> Result<Option<Region>, Cancelled> {
    if let Some(found) = vision_masks(sampler, area) {
        ctl.check()?;
        let union = found.union();
        if let Some(r) = found.region(&union) {
            return Ok(Some(r));
        }
    }
    subject::select_subject_with(sampler, area, ctl)
}

/// [`subject_region`] without cancellation.
pub(crate) fn subject_region_now(sampler: &dyn Sampler, area: Rect) -> Option<Region> {
    subject_region(sampler, area, &Interrupt::NONE).unwrap_or_default()
}

/// Object Selection's click: the Vision object under document point (`x`, `y`), else the Object
/// Finder's.
pub(crate) fn object_region_at(sampler: &dyn Sampler, area: Rect, x: f32, y: f32) -> Option<Region> {
    if let Some(found) = vision_masks(sampler, area)
        && let Some(r) = found.mask_at(x, y).and_then(|m| found.region(m))
    {
        return Some(r);
    }
    grabcut::object_at(sampler, area, x, y, grabcut::OBJECT_AT_WORK_PX)
}

/// Vision's object masks over `area` at the working size; `None` when Vision is off,
/// unavailable, failed or found nothing (the caller falls back).
fn vision_masks(sampler: &dyn Sampler, area: Rect) -> Option<Masks> {
    if !enabled() || area.is_empty() {
        return None;
    }
    let (w, h) = (area.width() as usize, area.height() as usize);
    let step = w.max(h).div_ceil(WORK_SIDE).max(1);
    let img = sampler.rgb_scaled(area, step);
    let rgba: Vec<u8> = img.px.iter().flat_map(|p| [p[0], p[1], p[2]].map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8).into_iter().chain([255])).collect();
    match photocraft_vision::foreground_instances(&rgba, img.w, img.h) {
        Ok(masks) if !masks.is_empty() => Some(Masks { area, step, w: img.w, h: img.h, masks }),
        Ok(_) => None,
        Err(e) => {
            eprintln!("warning: Apple Vision selection failed, using the built-in segmentation: {e}");
            None
        }
    }
}

/// Object masks at the working size (`w`×`h`, each pixel covering `step`² document pixels).
struct Masks {
    area: Rect,
    step: usize,
    w: usize,
    h: usize,
    masks: Vec<Vec<u8>>,
}

impl Masks {
    /// Every object at once (the largest coverage of each pixel).
    fn union(&self) -> Vec<u8> {
        (0..self.w * self.h).map(|i| self.masks.iter().filter_map(|m| m.get(i)).copied().max().unwrap_or(0)).collect()
    }

    /// The object that covers document point (`x`, `y`) most, if any covers it half or more.
    fn mask_at(&self, x: f32, y: f32) -> Option<&Vec<u8>> {
        let (lx, ly) = ((x - self.area.x0 as f32) / self.step as f32, (y - self.area.y0 as f32) / self.step as f32);
        if !(lx >= 0.0 && ly >= 0.0 && (lx as usize) < self.w && (ly as usize) < self.h) {
            return None;
        }
        let i = ly as usize * self.w + lx as usize;
        self.masks.iter().filter(|m| m.get(i).is_some_and(|&v| v >= 128)).max_by_key(|m| m.get(i).copied().unwrap_or(0))
    }

    /// `mask` scaled up to the document (bilinear) and trimmed to the pixels it covers.
    fn region(&self, mask: &[u8]) -> Option<Region> {
        upscale_region(mask, self.w, self.h, self.step, self.area)
    }
}

/// A `w`×`h` mask whose pixels each cover `step`² pixels of `area`, as a [`Region`] over `area`
/// at full resolution (bilinear between mask pixel centres), trimmed to its non-zero pixels.
/// `None` if nothing is covered or the mask doesn't match `w`×`h`.
fn upscale_region(mask: &[u8], w: usize, h: usize, step: usize, area: Rect) -> Option<Region> {
    if w == 0 || h == 0 || step == 0 || mask.len() != w.checked_mul(h)? {
        return None;
    }
    // Trim first at the working size (one mask pixel of margin for the bilinear fringe).
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for (i, _) in mask.iter().enumerate().filter(|(_, v)| **v > 0) {
        let (x, y) = (i % w, i / w);
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
    }
    if x0 >= x1 {
        return None;
    }
    let (aw, ah) = (area.width() as usize, area.height() as usize);
    let (dx0, dy0) = (x0.saturating_sub(1) * step, y0.saturating_sub(1) * step);
    let (dx1, dy1) = (((x1 + 1) * step).min(aw), ((y1 + 1) * step).min(ah));
    if dx0 >= dx1 || dy0 >= dy1 {
        return None;
    }
    let at = |x: usize, y: usize| f32::from(mask.get(y.min(h - 1) * w + x.min(w - 1)).copied().unwrap_or(0));
    let mut out = Vec::with_capacity((dx1 - dx0) * (dy1 - dy0));
    for dy in dy0..dy1 {
        let fy = ((dy as f32 + 0.5) / step as f32 - 0.5).max(0.0);
        let (ty, wy) = (fy as usize, fy.fract());
        for dx in dx0..dx1 {
            let fx = ((dx as f32 + 0.5) / step as f32 - 0.5).max(0.0);
            let (tx, wx) = (fx as usize, fx.fract());
            let top = at(tx, ty) * (1.0 - wx) + at(tx + 1, ty) * wx;
            let bottom = at(tx, ty + 1) * (1.0 - wx) + at(tx + 1, ty + 1) * wx;
            out.push((top * (1.0 - wy) + bottom * wy).round().clamp(0.0, 255.0) as u8);
        }
    }
    let bbox = Rect::new(area.x0 + dx0 as i32, area.y0 + dy0 as i32, area.x0 + dx1 as i32, area.y0 + dy1 as i32);
    Some(Region { bbox, mask: out })
}

#[cfg(test)]
mod tests {
    use super::{object_region_at, subject_region_now, upscale_region, vision_masks};
    use photocraft_algo::segment::{ImageSampler, RgbImage};
    use photocraft_geom::Rect;

    /// A `w`×`h` grey image with a red disc of radius `r` centred at (`cx`, `cy`).
    fn disc(w: usize, h: usize, (cx, cy, r): (f32, f32, f32)) -> RgbImage {
        let px = (0..w * h)
            .map(|i| {
                let (dx, dy) = ((i % w) as f32 + 0.5 - cx, (i / w) as f32 + 0.5 - cy);
                if dx * dx + dy * dy <= r * r { [0.86, 0.12, 0.12] } else { [0.5, 0.5, 0.5] }
            })
            .collect();
        RgbImage { w, h, px }
    }

    /// Where Vision runs (macOS 14+), Select Subject and a click take its object, in document
    /// coordinates, also on a document larger than the working size; a click on the background
    /// finds no Vision object. Elsewhere this checks nothing beyond not crashing.
    #[test]
    fn vision_objects_land_on_the_document() {
        for (w, h, disc_at) in [(320, 240, (160.0, 120.0, 60.0)), (6000, 4000, (3000.0, 2000.0, 1000.0))] {
            let img = disc(w, h, disc_at);
            let s = ImageSampler { img: &img, origin: (0, 0) };
            let area = Rect::new(0, 0, w as i32, h as i32);
            let subject = subject_region_now(&s, area);
            let Some(found) = vision_masks(&s, area) else {
                assert!(!photocraft_vision::available() || subject.is_some(), "the built-in segmentation still answers");
                continue;
            };
            let (cx, cy, r) = disc_at;
            let subject = subject.unwrap();
            assert!(subject.at(cx as i32, cy as i32) > 0.9, "{w}×{h}: the disc's centre is selected");
            assert!(subject.at(5, 5) < 0.1, "{w}×{h}: the corner is not");
            let b = subject.bbox;
            assert!((b.x0 as f32 - (cx - r)).abs() < r * 0.15 && (b.x1 as f32 - (cx + r)).abs() < r * 0.15, "{w}×{h}: bbox {b:?}");
            assert!(object_region_at(&s, area, cx, cy).is_some_and(|o| o.at(cx as i32, cy as i32) > 0.9));
            assert!(found.mask_at(5.0, 5.0).is_none(), "no Vision object on the background");
            assert!(found.mask_at(-1.0, 5.0).is_none() && found.mask_at(w as f32, 5.0).is_none(), "off the image is nothing");
        }
    }

    #[test]
    fn upscaling_trims_to_the_object_and_keeps_document_coordinates() {
        // 4×3 working mask, one covered pixel at (2, 1), each covering 10×10 document pixels.
        let mut mask = vec![0u8; 12];
        mask[4 + 2] = 255;
        let r = upscale_region(&mask, 4, 3, 10, Rect::new(100, 200, 140, 230)).unwrap();
        assert_eq!(r.bbox, Rect::new(110, 200, 140, 230));
        assert_eq!(r.mask.len(), 30 * 30);
        assert!(r.at(125, 215) > 0.9, "the covered pixel's centre is selected");
        assert!(r.at(135, 215) < r.at(125, 215), "the edge fades towards the next pixel");
        assert_eq!(r.at(111, 201), 0.0, "far corner of the margin is not");
    }

    #[test]
    fn empty_or_mismatched_masks_find_nothing() {
        let area = Rect::new(0, 0, 8, 8);
        assert!(upscale_region(&[0; 4], 2, 2, 4, area).is_none());
        assert!(upscale_region(&[255; 3], 2, 2, 4, area).is_none());
        assert!(upscale_region(&[255; 4], 2, 2, 0, area).is_none());
        assert!(upscale_region(&[], 0, 0, 4, area).is_none());
    }
}
