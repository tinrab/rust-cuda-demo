use std::ffi::c_int;

use crate::{error::CudaResult, handle_cuda_error};

pub mod ffi {
    use super::*;

    type CudaErrorType = c_int;

    #[link(name = "rustcudademo", kind = "static")]
    extern "C" {
        pub fn math_vector_add(
            n: c_int,
            a: *const f32,
            b: *const f32,
            c: *mut f32,
        ) -> CudaErrorType;
    }
}

pub fn vector_add(a: &[f32], b: &[f32]) -> CudaResult<Vec<f32>> {
    let mut c = vec![0.0; a.len()];
    unsafe {
        handle_cuda_error!(ffi::math_vector_add(
            a.len() as c_int,
            a.as_ptr(),
            b.as_ptr(),
            c.as_mut_ptr()
        ))?;
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let a = vec![3.0, 4.0, 2.0, -2.0];
        let b = vec![3.0, 5.0, 2.0, 4.0];
        assert_eq!(vector_add(&a, &b).unwrap(), vec![6.0, 9.0, 4.0, 2.0]);
    }
}
