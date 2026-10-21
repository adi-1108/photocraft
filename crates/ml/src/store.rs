//! Where the model files live. They are never committed or embedded: a build or a user puts them in
//! [`models_dir`] (see `docs/ml.md`).

use std::path::PathBuf;

/// A model the editor can run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Model {
    /// BiRefNet-lite, 1024 px salient-object segmentation (MIT).
    BirefnetLite,
    /// big-lama, 512 px inpainting (Apache-2.0).
    Lama,
}

impl Model {
    /// All supported models, in display/download order.
    pub const ALL: [Self; 2] = [Self::BirefnetLite, Self::Lama];

    /// Published ONNX download.
    pub fn url(self) -> &'static str {
        match self {
            Self::BirefnetLite => "https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/model.onnx",
            Self::Lama => "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
        }
    }

    /// SHA-256 of the published file, measured from the installed copies with `shasum -a 256`.
    pub fn sha256(self) -> &'static str {
        match self {
            Self::BirefnetLite => "5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333",
            Self::Lama => "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6",
        }
    }

    /// Licence of the model weights.
    pub fn licence(self) -> &'static str {
        match self {
            Self::BirefnetLite => "MIT",
            Self::Lama => "Apache-2.0",
        }
    }

    /// File name inside the models folder.
    pub fn file_name(self) -> &'static str {
        match self {
            Self::BirefnetLite => "birefnet_lite.onnx",
            Self::Lama => "lama_fp32.onnx",
        }
    }

    /// Human name for messages.
    pub fn label(self) -> &'static str {
        match self {
            Self::BirefnetLite => "subject segmentation (BiRefNet-lite)",
            Self::Lama => "inpainting (LaMa)",
        }
    }

    /// Size of the published file, to catch a truncated download.
    pub fn size_bytes(self) -> u64 {
        match self {
            Self::BirefnetLite => 224_005_088,
            Self::Lama => 208_044_816,
        }
    }
}

/// The folder models are read from: `PHOTOCRAFT_MODELS_DIR`, else the per-user data folder
/// (`~/Library/Application Support/PhotoCraft/models`, `$XDG_DATA_HOME/photocraft/models`,
/// `%APPDATA%\PhotoCraft\models`). `None` when no home folder is known (or in the browser).
pub fn models_dir() -> Option<PathBuf> {
    if cfg!(target_arch = "wasm32") {
        return None;
    }
    if let Some(d) = std::env::var_os("PHOTOCRAFT_MODELS_DIR").filter(|d| !d.is_empty()) {
        return Some(PathBuf::from(d));
    }
    let var = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    if cfg!(target_os = "macos") {
        var("HOME").map(|h| h.join("Library").join("Application Support").join("PhotoCraft").join("models"))
    } else if cfg!(windows) {
        var("APPDATA").map(|a| a.join("PhotoCraft").join("models"))
    } else {
        var("XDG_DATA_HOME").or_else(|| var("HOME").map(|h| h.join(".local").join("share"))).map(|d| d.join("photocraft").join("models"))
    }
}

static MODELS_DISABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Treat every model as missing in this process from now on (tests, fuzzing), so results never
/// depend on what a developer has installed. `PHOTOCRAFT_NO_MODELS` does the same for a shell.
pub fn disable_models() {
    MODELS_DISABLED.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Whether models are switched off by [`disable_models`] or `PHOTOCRAFT_NO_MODELS`.
pub fn models_disabled() -> bool {
    MODELS_DISABLED.load(std::sync::atomic::Ordering::Relaxed) || std::env::var_os("PHOTOCRAFT_NO_MODELS").is_some_and(|v| !v.is_empty() && v != "0")
}

/// The installed file of `model`, if it is there and the right size. A partial download is treated
/// as missing rather than handed to the loader. `None` while [`models_disabled`].
pub fn model_path(model: Model) -> Option<PathBuf> {
    if models_disabled() {
        return None;
    }
    let p = models_dir()?.join(model.file_name());
    let len = std::fs::metadata(&p).ok()?.len();
    (len == model.size_bytes()).then_some(p)
}

/// Where `model` is expected, installed or not (for messages).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn expected_path(model: Model) -> String {
    models_dir().map_or_else(|| model.file_name().to_string(), |d| d.join(model.file_name()).display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_models_are_missing() {
        disable_models();
        assert!(models_disabled());
        for model in Model::ALL {
            assert_eq!(model_path(model), None);
        }
    }
}
