use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Linking paths are dependent on your Linux distribution.
    // Be aware of LIBRARY_PATH and LD_LIBRARY_PATH.

    cc::Build::new()
        .cuda(true)
        .cudart("static")
        .includes(&["./cuda", "/opt/cuda/include"])
        .files(&[
            "./cuda/lib.cu",
            "./cuda/lib_math.cu",
            "./cuda/lib_cublas.cu",
        ])
        // Needed because nvcc requires specific gcc version.
        .flag("-ccbin=gcc-13")
        .warnings(false)
        .extra_warnings(false)
        .compile("rustcudademo");

    println!("cargo:rustc-link-search=native=/opt/cuda/lib");

    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");

    println!("cargo:rerun-if-changed={}", "./build.rs");
    println!("cargo:rerun-if-changed={}", "./cuda");

    Ok(())
}