use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=./");
    Command::new("./cargo-git-version.sh")
        .arg(env!("CARGO_PKG_VERSION"))
        .spawn()
        .expect("Failed to update git info");
}
