//! The tract plumbing: load a model once per process, run it on flat `f32` tensors.

use std::sync::{Arc, Mutex, PoisonError};

use tract_onnx::prelude::*;

use crate::store::{Model, expected_path, model_path};
use crate::{Cancel, MlError, check_cancel};

type Plan = Arc<TypedRunnableModel>;

static PLANS: Mutex<Vec<(Model, Plan)>> = Mutex::new(Vec::new());
/// One inference at a time: a cancelled job's worker may still be inside tract (which can't be
/// interrupted), so a retry waits for it instead of stacking a second full-size run in memory.
static RUNNING: Mutex<()> = Mutex::new(());

fn backend<E: std::fmt::Display>(e: E) -> MlError {
    MlError::Backend(e.to_string())
}

/// The optimised plan of `model`, loaded on first use (a few seconds) and kept for the process
/// while the file stays installed: a removed or disabled model is dropped, not served from cache.
fn plan(model: Model) -> Result<Plan, MlError> {
    let mut plans = PLANS.lock().unwrap_or_else(PoisonError::into_inner);
    let Some(path) = model_path(model) else {
        plans.retain(|(m, _)| *m != model);
        return Err(MlError::ModelMissing { model, path: expected_path(model) });
    };
    if let Some((_, p)) = plans.iter().find(|(m, _)| *m == model) {
        return Ok(Arc::clone(p));
    }
    let plan = tract_onnx::onnx().model_for_path(path).and_then(|m| m.into_optimized()).and_then(|m| m.into_runnable()).map_err(backend)?;
    plans.push((model, Arc::clone(&plan)));
    Ok(plan)
}

/// Runs `model` on `inputs` (shape, row-major data) and returns its first output's data. Stops with [`MlError::Cancelled`] after loading the model and after waiting for
/// another inference to finish. tract can't be interrupted mid-run, so that is the last check.
pub(crate) fn run_with(model: Model, inputs: &[(&[usize], Vec<f32>)], cancel: Cancel<'_>) -> Result<Vec<f32>, MlError> {
    let plan = plan(model)?;
    check_cancel(cancel)?;
    let _one = RUNNING.lock().unwrap_or_else(PoisonError::into_inner);
    check_cancel(cancel)?;
    let mut feed: TVec<TValue> = TVec::new();
    for (shape, data) in inputs {
        feed.push(Tensor::from_shape(shape, data).map_err(backend)?.into());
    }
    let out = plan.run(feed).map_err(backend)?;
    let first = out.first().ok_or_else(|| MlError::Backend("the model returned no output".into()))?;
    let view = first.to_plain_array_view::<f32>().map_err(backend)?;
    Ok(view.iter().copied().collect())
}
