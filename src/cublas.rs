use std::ffi::c_int;

use crate::error::CudaResult;

pub mod ffi {
    use std::ffi::c_char;

    use super::*;

    type CudaErrorType = c_int;

    type CublasStatusType = c_int;

    #[link(name = "rustcudademo", kind = "static")]
    extern "C" {
        pub fn cublas_sgemm(
            n: c_int,
            a_matrix: *const f32,
            b_matrix: *const f32,
            c_matrix: *mut f32,
            alpha: f32,
            beta: f32,
            cublas_status: *mut CublasStatusType,
        ) -> CudaErrorType;

        // This function returns the string representation of a given status.
        #[link_name = "cublasGetStatusName"]
        pub fn cublas_get_status_name(status: CublasStatusType) -> *const c_char;

        // This function returns the description string for a given status.
        #[link_name = "cublasGetStatusString"]
        pub fn cublas_get_status_string(status: CublasStatusType) -> *const c_char;
    }
}

macro_rules! handle_cublas_error {
    ($expr:expr, $status:expr) => {{
        use $crate::error::{CublasError, CudaResult, InternalError};
        let error_code = { $expr };
        if ($status) != 0 {
            CudaResult::Err(CublasError::new($status).into())
        } else if error_code != 0 {
            CudaResult::Err(InternalError::new(error_code).into())
        } else {
            CudaResult::Ok(())
        }
    }};
}

pub fn sgemm(n: usize, a: &[f32], b: &[f32], alpha: f32, beta: f32) -> CudaResult<Vec<f32>> {
    let mut c = vec![0.0; n * n];
    unsafe {
        let mut cublas_status = 0;
        handle_cublas_error!(
            ffi::cublas_sgemm(
                n as c_int,
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                alpha,
                beta,
                &mut cublas_status,
            ),
            cublas_status
        )?;
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = sgemm(2, &a, &b, 1.0, 0.0).unwrap();
        assert_eq!(c, vec![7.0, 10.0, 15.0, 22.0]);
    }
}
