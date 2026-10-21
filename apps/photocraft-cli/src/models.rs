//! `photocraft-cli models list|fetch`: the command-line face of `photocraft_ml::download`
//! (Settings › AI Models is the other).

use std::io::Write;

use photocraft_ml::{Model, model_path, models_dir};

use crate::{Args, R};

#[derive(Debug, PartialEq, Eq)]
enum Action {
    List,
    Fetch(Vec<Model>),
}

fn action(args: &Args) -> Result<Action, String> {
    let [verb] = args.positional.as_slice() else {
        return Err("models needs list or fetch [--model birefnet|lama|all]".into());
    };
    match verb.as_str() {
        "list" if args.flags.is_empty() => Ok(Action::List),
        "list" => Err("models list takes no flags".into()),
        "fetch" => {
            if args.flags.len() > 1 {
                return Err("models fetch takes --model only once".into());
            }
            let models = match args.get("--model").unwrap_or("all") {
                "all" => Model::ALL.to_vec(),
                "birefnet" => vec![Model::BirefnetLite],
                "lama" => vec![Model::Lama],
                value => return Err(format!("bad --model `{value}`: expected birefnet, lama or all")),
            };
            Ok(Action::Fetch(models))
        }
        _ => Err(format!("unknown models command `{verb}`: expected list or fetch")),
    }
}

pub(super) fn run(args: &Args, out: &mut dyn Write) -> R {
    match action(args)? {
        Action::List => {
            let dir = models_dir();
            for model in Model::ALL {
                let path = dir
                    .as_ref()
                    .map_or_else(|| format!("{} (models directory unavailable)", model.file_name()), |dir| dir.join(model.file_name()).display().to_string());
                let installed = if model_path(model).is_some() { "yes" } else { "no" };
                writeln!(out, "{}\t{}\t{path}\tinstalled: {installed}", model.label(), model.licence()).map_err(|e| e.to_string())?;
            }
        }
        Action::Fetch(models) => {
            for model in models {
                // Progress goes to stderr in 10% steps, so stdout stays one line per model.
                let shown = std::cell::Cell::new(0u64);
                let progress = |done: u64, total: u64| {
                    let pct = done.saturating_mul(100).checked_div(total).unwrap_or(0);
                    if pct >= shown.get() + 10 || (done == total && shown.get() < 100) {
                        shown.set(pct);
                        eprintln!("{}: {pct}%", model.label());
                    }
                };
                let path = photocraft_ml::download(model, &progress, &|| false).map_err(|e| format!("{}: {e}", model.label()))?;
                writeln!(out, "Installed {} at {}", model.label(), path.display()).map_err(|e| e.to_string())?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(words: &[&str]) -> Result<Action, String> {
        let sub = crate::SUBCOMMANDS.iter().find(|s| s.name == "models").ok_or("missing subcommand")?;
        let words = words.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        action(&crate::parse(sub, &words)?.ok_or("help requested")?)
    }

    #[test]
    fn parses_models_actions() -> R {
        assert_eq!(parse(&["list"])?, Action::List);
        for words in [&["fetch"][..], &["fetch", "--model", "all"][..]] {
            assert_eq!(parse(words)?, Action::Fetch(Model::ALL.to_vec()));
        }
        assert_eq!(parse(&["fetch", "--model", "birefnet"])?, Action::Fetch(vec![Model::BirefnetLite]));
        assert_eq!(parse(&["fetch", "--model=lama"])?, Action::Fetch(vec![Model::Lama]));
        Ok(())
    }

    #[test]
    fn rejects_invalid_arguments_without_network() {
        for words in [
            vec![],
            vec!["unknown"],
            vec!["fetch", "extra"],
            vec!["list", "--model", "all"],
            vec!["fetch", "--model"],
            vec!["fetch", "--model", "other"],
            vec!["fetch", "--model="],
            vec!["fetch", "--unknown"],
            vec!["fetch", "--model", "lama", "--model", "all"],
        ] {
            assert!(parse(&words).is_err(), "{words:?}");
        }
    }

    #[test]
    fn list_reports_model_metadata_and_loader_status() -> R {
        let mut out = Vec::new();
        run(&Args { positional: vec!["list".into()], flags: vec![] }, &mut out)?;
        let output = String::from_utf8(out).map_err(|e| e.to_string())?;
        assert_eq!(output.lines().count(), Model::ALL.len());
        for model in Model::ALL {
            let status = if model_path(model).is_some() { "yes" } else { "no" };
            let line = output.lines().find(|line| line.contains(model.label())).ok_or("missing model")?;
            assert!(line.contains(model.licence()));
            assert!(line.contains(model.file_name()));
            assert!(line.ends_with(&format!("installed: {status}")));
        }
        Ok(())
    }
}
