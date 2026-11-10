use super::{Error, MAX_SIDE, available, foreground_instances};

/// A `w`×`h` grey image with a saturated red disc of radius `r` at the centre.
fn disc(w: usize, h: usize, r: f32) -> Vec<u8> {
    let mut px = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = (x as f32 + 0.5 - w as f32 / 2.0, y as f32 + 0.5 - h as f32 / 2.0);
            px.extend_from_slice(if dx * dx + dy * dy <= r * r { &[220, 30, 30, 255] } else { &[128, 128, 128, 255] });
        }
    }
    px
}

#[test]
fn bad_sizes_are_errors_not_crashes() {
    assert_eq!(foreground_instances(&[], 0, 0), Err(Error::BadSize));
    assert_eq!(foreground_instances(&[0; 4], 1, 0), Err(Error::BadSize));
    assert_eq!(foreground_instances(&[0; 15], 2, 2), Err(Error::BadSize));
    assert_eq!(foreground_instances(&[0; 4], usize::MAX, usize::MAX), Err(Error::BadSize));
    assert_eq!(foreground_instances(&[0; 4], MAX_SIDE + 1, 1), Err(Error::BadSize));
}

#[test]
fn masks_match_the_image_size() {
    let (w, h) = (320, 240);
    let found = foreground_instances(&disc(w, h, 60.0), w, h).unwrap();
    assert!(found.iter().all(|m| m.len() == w * h));
    if !available() {
        assert!(found.is_empty(), "nothing is found without Vision");
    }
}

#[test]
fn a_flat_image_has_no_foreground() {
    let (w, h) = (64, 48);
    let flat: Vec<u8> = [128, 128, 128, 255].repeat(w * h);
    let found = foreground_instances(&flat, w, h).unwrap();
    assert!(found.iter().all(|m| m.iter().all(|&v| v < 128)), "a flat image has no object");
}

/// On macOS 14+, Vision finds the disc and not the background around it.
#[test]
fn vision_finds_a_centred_object() {
    if !available() {
        return;
    }
    let (w, h) = (320, 240);
    let found = foreground_instances(&disc(w, h, 60.0), w, h).unwrap();
    let union: Vec<u8> = (0..w * h).map(|i| found.iter().map(|m| m[i]).max().unwrap_or(0)).collect();
    assert!(union[(h / 2) * w + w / 2] >= 128, "the disc's centre is selected");
    assert!(union[5 * w + 5] < 128, "the corner is not");
}
