use std::fs;
use std::path::PathBuf;

fn terminate_ui_expression(
    source: &mut String,
    signature: &str,
    indentation: &str,
) -> Result<bool, String> {
    let start = source
        .find(signature)
        .ok_or_else(|| format!("could not find method signature: {signature}"))?;
    let body_start = start + signature.len();
    let tail = &source[body_start..];
    let body_end = [tail.find("\n    fn "), tail.find("\n}\n\nimpl ")]
        .into_iter()
        .flatten()
        .min()
        .map_or(source.len(), |offset| body_start + offset);
    let closing = format!("\n{indentation}}})");
    let relative = source[body_start..body_end]
        .rfind(&closing)
        .ok_or_else(|| format!("could not find trailing UI expression in: {signature}"))?;
    let insertion = body_start + relative + closing.len();

    if source.as_bytes().get(insertion) == Some(&b';') {
        return Ok(false);
    }

    source.insert(insertion, ';');
    Ok(true)
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/main.rs");

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
    let mut source = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    let mut changed = false;

    for (signature, indentation) in [
        (
            "fn legend(&mut self, ui: &mut egui::Ui, p: &PresentationProjection)",
            "        ",
        ),
        (
            "fn summary(&self, ui: &mut egui::Ui, p: &PresentationProjection)",
            "        ",
        ),
        (
            "fn settings(&mut self, ui: &mut egui::Ui, p: &PresentationProjection)",
            "            ",
        ),
    ] {
        changed |= terminate_ui_expression(&mut source, signature, indentation)
            .unwrap_or_else(|error| panic!("{error}"));
    }

    if changed {
        fs::write(&path, source)
            .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
    }
}
