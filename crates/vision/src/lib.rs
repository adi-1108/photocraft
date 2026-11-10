//! Foreground object masks from Apple's Vision framework, for Select Subject, Object Selection
//! and Remove Background on macOS.
//!
//! - **macOS 14+** ([`foreground_instances`]): `VNGenerateForegroundInstanceMaskRequest` finds
//!   the salient foreground objects (people, animals, things) on device, one soft mask each.
//!   Calling Vision and CoreVideo needs Objective-C interop, so `macos.rs` allows `unsafe` (an
//!   isolated helper crate, like `photocraft-tablet`, with a `SAFETY:` comment on every block).
//! - **Everywhere else** (older macOS, Linux, Windows, the web): nothing is found, and callers
//!   fall back to their own segmentation. [`available`] says which applies.
//!
//! Masks are `width * height` bytes, row-major, `0` = background, `255` = object, soft at edges.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

#[cfg(target_os = "macos")]
mod macos;

/// Longest side Vision is given, in pixels: it works on a small internal image anyway, and the
/// cap bounds the RGBA copy and every returned mask (`MAX_SIDE²` bytes each).
pub const MAX_SIDE: usize = 4096;

/// Most objects returned; Vision rarely finds more than a handful.
pub const MAX_INSTANCES: usize = 32;

/// Why Vision gave no answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The pixels don't match `width * height * 4`, or a side is 0 or over [`MAX_SIDE`].
    BadSize,
    /// Vision or CoreVideo reported a failure.
    Vision(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadSize => write!(f, "image size doesn't match its pixels or is out of range (1..={MAX_SIDE} per side)"),
            Self::Vision(m) => write!(f, "Apple Vision: {m}"),
        }
    }
}

impl std::error::Error for Error {}

/// Whether this machine can run Vision's foreground masks (macOS 14 or later).
pub fn available() -> bool {
    #[cfg(target_os = "macos")]
    {
        macos::available()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// One soft mask per foreground object Vision finds in `rgba` (straight 8-bit RGBA, row-major,
/// `width * height * 4` bytes), each `width * height` bytes. Empty when nothing stands out or
/// Vision isn't [`available`].
pub fn foreground_instances(rgba: &[u8], width: usize, height: usize) -> Result<Vec<Vec<u8>>, Error> {
    check_size(rgba.len(), width, height)?;
    #[cfg(target_os = "macos")]
    {
        if macos::available() { macos::foreground_instances(rgba, width, height) } else { Ok(Vec::new()) }
    }
    #[cfg(not(target_os = "macos"))]
    {
        Ok(Vec::new())
    }
}

fn check_size(len: usize, width: usize, height: usize) -> Result<(), Error> {
    let in_range = (1..=MAX_SIDE).contains(&width) && (1..=MAX_SIDE).contains(&height);
    if in_range && width.checked_mul(height).and_then(|n| n.checked_mul(4)) == Some(len) { Ok(()) } else { Err(Error::BadSize) }
}

#[cfg(test)]
mod tests;
