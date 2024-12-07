use std::{
    ffi::CStr,
    fmt::{self, Display},
};

use crate::{cublas, cuda};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaError {
    Internal(InternalError),
    Cublas(CublasError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InternalError {
    code: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CublasError {
    pub status: i32,
}

pub type CudaResult<T> = Result<T, CudaError>;

impl Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CudaError::Internal(error) => write!(f, "internal: {}", error),
            CudaError::Cublas(error) => write!(f, "cublas: {}", error),
        }
    }
}

impl InternalError {
    pub fn new(code: i32) -> Self {
        Self { code }
    }

    pub fn code(&self) -> i32 {
        self.code
    }

    pub fn message(&self) -> &str {
        unsafe {
            CStr::from_ptr(cuda::ffi::cuda_get_error_string(self.code))
                .to_str()
                .unwrap()
        }
    }
}

impl From<InternalError> for CudaError {
    fn from(error: InternalError) -> Self {
        CudaError::Internal(error)
    }
}

impl Display for InternalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "InternalError({}): {}", self.code(), self.message())
    }
}

impl CublasError {
    pub fn new(status: i32) -> Self {
        Self { status }
    }

    pub fn status(&self) -> i32 {
        self.status
    }

    pub fn name(&self) -> &str {
        unsafe {
            CStr::from_ptr(cublas::ffi::cublas_get_status_name(self.status))
                .to_str()
                .unwrap()
        }
    }

    pub fn message(&self) -> &str {
        unsafe {
            CStr::from_ptr(cublas::ffi::cublas_get_status_string(self.status))
                .to_str()
                .unwrap()
        }
    }
}

impl From<CublasError> for CudaError {
    fn from(error: CublasError) -> Self {
        CudaError::Cublas(error)
    }
}

impl Display for CublasError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "CublasError({}): {} - {}",
            self.status(),
            self.name(),
            self.message()
        )
    }
}
