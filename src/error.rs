use std::fmt::{self, Display};

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CudaError {
    #[error("internal: {0}")]
    Internal(InternalError),
    #[error("cublas: {0}")]
    Cublas(CublasError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InternalError {
    pub id: i32,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CublasError {
    pub id: i32,
    pub name: String,
    pub message: String,
}

pub type CudaResult<T> = Result<T, CudaError>;

impl From<InternalError> for CudaError {
    fn from(error: InternalError) -> Self {
        CudaError::Internal(error)
    }
}

impl Display for InternalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "InternalError({}): {}", self.id, self.message)
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
            self.id, self.name, self.message
        )
    }
}
