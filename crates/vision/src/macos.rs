//! Vision's foreground instance masks through objc2. Every `unsafe` block calls a Vision or
//! CoreVideo API with arguments that meet its documented contract (each has a `SAFETY:` comment);
//! all pixel copies check their bounds against the buffer's own width, height and row stride.
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::AnyThread;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::AnyClass;
use objc2_core_video::{
    CVPixelBuffer, CVPixelBufferCreate, CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow, CVPixelBufferGetHeight, CVPixelBufferGetPixelFormatType,
    CVPixelBufferGetWidth, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress, kCVPixelFormatType_32BGRA,
    kCVPixelFormatType_OneComponent32Float, kCVReturnSuccess,
};
use objc2_foundation::{NSArray, NSDictionary, NSIndexSet, NSNotFound};
use objc2_vision::{VNGenerateForegroundInstanceMaskRequest, VNImageRequestHandler, VNRequest};

use crate::{Error, MAX_INSTANCES};

/// The request class exists from macOS 14; on older systems the runtime has no such class.
pub(crate) fn available() -> bool {
    AnyClass::get(c"VNGenerateForegroundInstanceMaskRequest").is_some()
}

pub(crate) fn foreground_instances(rgba: &[u8], width: usize, height: usize) -> Result<Vec<Vec<u8>>, Error> {
    // Vision autoreleases; worker threads have no pool of their own.
    autoreleasepool(|_| run(rgba, width, height))
}

fn run(rgba: &[u8], width: usize, height: usize) -> Result<Vec<Vec<u8>>, Error> {
    let image = bgra_buffer(rgba, width, height)?;
    let options = NSDictionary::new();
    // SAFETY: `image` is a valid, unlocked 32BGRA pixel buffer we own; `options` is an empty
    // dictionary of the documented key/value types; `alloc` gives a fresh handler to initialise.
    let handler = unsafe { VNImageRequestHandler::initWithCVPixelBuffer_options(VNImageRequestHandler::alloc(), &image, &options) };
    // SAFETY: plain `new`; the class exists (checked by `available` before this runs).
    let request = unsafe { VNGenerateForegroundInstanceMaskRequest::new() };
    let as_request: Retained<VNRequest> = Retained::into_super(Retained::into_super(request.clone()));
    let requests = NSArray::from_retained_slice(&[as_request]);
    handler.performRequests_error(&requests).map_err(|e| Error::Vision(e.localizedDescription().to_string()))?;
    // SAFETY: `results` is read after `performRequests` returned on this thread.
    let Some(observation) = (unsafe { request.results() }).and_then(|r| r.firstObject()) else {
        return Ok(Vec::new());
    };
    // SAFETY: an observation from a finished request; `allInstances` has no preconditions.
    let all = unsafe { observation.allInstances() };
    let mut masks = Vec::new();
    let mut index = all.firstIndex();
    while index != NSNotFound as usize && masks.len() < MAX_INSTANCES {
        let one = NSIndexSet::indexSetWithIndex(index);
        // SAFETY: `one` holds an index from this observation's own `allInstances`, and `handler`
        // is the handler that performed the request, as the method requires.
        let scaled = unsafe { observation.generateScaledMaskForImageForInstances_fromRequestHandler_error(&one, &handler) }
            .map_err(|e| Error::Vision(e.localizedDescription().to_string()))?;
        masks.push(read_float_mask(&scaled, width, height)?);
        index = all.indexGreaterThanIndex(index);
    }
    Ok(masks)
}

/// A new 32BGRA pixel buffer holding `rgba` (RGBA → BGRA, row by row at the buffer's stride).
fn bgra_buffer(rgba: &[u8], width: usize, height: usize) -> Result<Retained<CVPixelBuffer>, Error> {
    let mut out: *mut CVPixelBuffer = std::ptr::null_mut();
    // SAFETY: `out` is a valid place for the created buffer; no allocator or attributes (defaults).
    let status = unsafe { CVPixelBufferCreate(None, width, height, kCVPixelFormatType_32BGRA, None, NonNull::from(&mut out)) };
    if status != kCVReturnSuccess {
        return Err(Error::Vision(format!("CVPixelBufferCreate failed ({status})")));
    }
    let ptr = NonNull::new(out).ok_or_else(|| Error::Vision("CVPixelBufferCreate returned no buffer".into()))?;
    // SAFETY: a successful Create returns a +1 reference we now own (the Create rule).
    let buffer = unsafe { Retained::from_raw(ptr.as_ptr()) }.ok_or_else(|| Error::Vision("no buffer".into()))?;
    with_locked(&buffer, CVPixelBufferLockFlags::empty(), |base, stride, bw, bh| {
        if bw < width || bh < height || stride < width * 4 {
            return Err(Error::Vision("pixel buffer smaller than the image".into()));
        }
        for y in 0..height {
            let src = rgba.get(y * width * 4..(y + 1) * width * 4).ok_or(Error::BadSize)?;
            // SAFETY: the buffer is locked for writing, `y < bh` and `width * 4 <= stride`, so
            // the row lies inside the buffer's allocation of `stride * bh` bytes.
            let dst = unsafe { std::slice::from_raw_parts_mut(base.add(y * stride), width * 4) };
            for (d, s) in dst.as_chunks_mut::<4>().0.iter_mut().zip(src.as_chunks::<4>().0) {
                *d = [s[2], s[1], s[0], s[3]];
            }
        }
        Ok(())
    })?;
    Ok(buffer)
}

/// A one-component float mask at the image size, as bytes (`0.0..=1.0` → `0..=255`).
fn read_float_mask(buffer: &CVPixelBuffer, width: usize, height: usize) -> Result<Vec<u8>, Error> {
    if CVPixelBufferGetPixelFormatType(buffer) != kCVPixelFormatType_OneComponent32Float {
        return Err(Error::Vision("unexpected mask format".into()));
    }
    with_locked(buffer, CVPixelBufferLockFlags::ReadOnly, |base, stride, bw, bh| {
        if bw != width || bh != height || stride < width * 4 || base.align_offset(std::mem::align_of::<f32>()) != 0 {
            return Err(Error::Vision(format!("mask is {bw}×{bh}, expected {width}×{height}")));
        }
        let mut mask = Vec::with_capacity(width * height);
        for y in 0..height {
            // SAFETY: locked for reading; `y < bh`, `width * 4 <= stride` and `base` is f32-aligned
            // (checked above; every row starts at a multiple of the stride, which CoreVideo
            // aligns), so the row of `width` floats is inside the allocation.
            let row = unsafe { std::slice::from_raw_parts(base.add(y * stride).cast::<f32>(), width) };
            mask.extend(row.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8));
        }
        Ok(mask)
    })
}

/// Runs `f(base, bytes_per_row, width, height)` with `buffer`'s pixels locked, unlocking after.
fn with_locked<R>(
    buffer: &CVPixelBuffer,
    flags: CVPixelBufferLockFlags,
    f: impl FnOnce(*mut u8, usize, usize, usize) -> Result<R, Error>,
) -> Result<R, Error> {
    // SAFETY: a valid buffer; every successful lock is paired with the unlock below.
    if unsafe { CVPixelBufferLockBaseAddress(buffer, flags) } != kCVReturnSuccess {
        return Err(Error::Vision("could not lock the pixel buffer".into()));
    }
    let base: *mut c_void = CVPixelBufferGetBaseAddress(buffer);
    let out = if base.is_null() {
        Err(Error::Vision("pixel buffer has no memory".into()))
    } else {
        f(base.cast::<u8>(), CVPixelBufferGetBytesPerRow(buffer), CVPixelBufferGetWidth(buffer), CVPixelBufferGetHeight(buffer))
    };
    // SAFETY: unlocks the lock taken above with the same flags.
    unsafe { CVPixelBufferUnlockBaseAddress(buffer, flags) };
    out
}
