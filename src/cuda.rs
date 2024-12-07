use crate::error::CudaResult;

pub mod ffi {
    use std::ffi::{c_char, c_int};

    type CudaErrorType = c_int;

    #[link(name = "rustcudademo", kind = "static")]
    extern "C" {
        /// Returns which device is currently being used.
        #[link_name = "cudaGetDevice"]
        pub fn cuda_get_device(device: &mut c_int) -> CudaErrorType;

        /// Returns the number of compute-capable devices.
        #[link_name = "cudaGetDeviceCount"]
        pub fn cuda_get_device_count(count: &mut c_int) -> CudaErrorType;

        /// Returns the description string for an error code.
        #[link_name = "cudaGetErrorString"]
        pub fn cuda_get_error_string(code: c_int) -> *const c_char;
    }
}

#[macro_export]
macro_rules! handle_cuda_error {
    ($expr:expr) => {{
        use $crate::error::{CudaResult, InternalError};
        let code = { $expr };
        if code != 0 {
            CudaResult::Err(InternalError::new(code).into())
        } else {
            CudaResult::Ok(())
        }
    }};
}

pub fn get_device_count() -> CudaResult<u32> {
    let mut count = 0;
    unsafe {
        handle_cuda_error!(ffi::cuda_get_device_count(&mut count))?;
    }
    Ok(count as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let device_count = get_device_count().unwrap();
        assert_ne!(device_count, 0);
    }
}
