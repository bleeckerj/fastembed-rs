//! Initialization options for the text embedding models.
//!
use std::fmt::{self, Debug, Formatter};

use crate::{
    common::{TokenizerFiles, DEFAULT_CACHE_DIR},
    pooling::Pooling,
    EmbeddingModel, QuantizationMode,
};
use ort::{execution_providers::ExecutionProviderDispatch, session::Session};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

use super::{DEFAULT_EMBEDDING_MODEL, DEFAULT_MAX_LENGTH};

/// Wrapper type for values that don't implement Debug
#[derive(Clone)]
pub struct DebugIgnored<T>(pub T);

impl<T> Debug for DebugIgnored<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "<custom progress>")
    }
}

/// Options for initializing the TextEmbedding model
/// 
pub struct InitOptions {
    pub model_name: EmbeddingModel,
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub max_length: usize,
    pub cache_dir: PathBuf,
    pub show_download_progress: bool,
    pub custom_progress: Option<Box<dyn hf_hub::api::Progress + Send + Sync + 'static>>,
}

// Manual Debug implementation
impl std::fmt::Debug for InitOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InitOptions")
            .field("model_name", &self.model_name)
            .field("execution_providers", &self.execution_providers)
            .field("max_length", &self.max_length)
            .field("cache_dir", &self.cache_dir)
            .field("show_download_progress", &self.show_download_progress)
            .field("custom_progress", &if self.custom_progress.is_some() { "Some(<progress>)" } else { "None" })
            .finish()
    }
}

// Manual Clone implementation
impl Clone for InitOptions {
    fn clone(&self) -> Self {
        Self {
            model_name: self.model_name.clone(),
            execution_providers: self.execution_providers.clone(),
            max_length: self.max_length,
            cache_dir: self.cache_dir.clone(),
            show_download_progress: self.show_download_progress,
            custom_progress: None, // Progress can't be cloned
        }
    }
}

impl InitOptions {
    // Add this method
    pub fn with_custom_progress<P>(mut self, progress: P) -> Self 
    where P: hf_hub::api::Progress + Send + Sync + 'static 
    {
        self.custom_progress = Some(Box::new(progress));
        // Set show_download_progress to false to avoid conflicts
        self.show_download_progress = false;
        self
    }
    /// Create a new InitOptions with the given model name
    pub fn new(model_name: EmbeddingModel) -> Self {
        Self {
            model_name,
            ..Default::default()
        }
    }
    
    /// Set the maximum length of the input text
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }
    
    /// Set the cache directory for the model files
    pub fn with_cache_dir(mut self, cache_dir: PathBuf) -> Self {
        self.cache_dir = cache_dir;
        self
    }
    
    /// Set the execution providers for the model
    pub fn with_execution_providers(
        mut self,
        execution_providers: Vec<ExecutionProviderDispatch>,
    ) -> Self {
        self.execution_providers = execution_providers;
        self
    }
    
    /// Set whether to show download progress
    pub fn with_show_download_progress(mut self, show_download_progress: bool) -> Self {
        self.show_download_progress = show_download_progress;
        self
    }
}

impl Default for InitOptions {
    fn default() -> Self {
        Self {
            model_name: DEFAULT_EMBEDDING_MODEL,
            execution_providers: Default::default(),
            max_length: DEFAULT_MAX_LENGTH,
            cache_dir: Path::new(DEFAULT_CACHE_DIR).to_path_buf(),
            show_download_progress: true,
            custom_progress: None,
        }
    }
}

/// Options for initializing UserDefinedEmbeddingModel
///
/// Model files are held by the UserDefinedEmbeddingModel struct
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct InitOptionsUserDefined {
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub max_length: usize,
}

impl InitOptionsUserDefined {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }
    
    pub fn with_execution_providers(
        mut self,
        execution_providers: Vec<ExecutionProviderDispatch>,
    ) -> Self {
        self.execution_providers = execution_providers;
        self
    }
    
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }
}

impl Default for InitOptionsUserDefined {
    fn default() -> Self {
        Self {
            execution_providers: Default::default(),
            max_length: DEFAULT_MAX_LENGTH,
        }
    }
}

/// Convert InitOptions to InitOptionsUserDefined
///
/// This is useful for when the user wants to use the same options for both the default and user-defined models
impl From<InitOptions> for InitOptionsUserDefined {
    fn from(options: InitOptions) -> Self {
        InitOptionsUserDefined {
            execution_providers: options.execution_providers,
            max_length: options.max_length,
        }
    }
}

/// Struct for "bring your own" embedding models
///
/// The onnx_file and tokenizer_files are expecting the files' bytes
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct UserDefinedEmbeddingModel {
    pub onnx_file: Vec<u8>,
    pub tokenizer_files: TokenizerFiles,
    pub pooling: Option<Pooling>,
    pub quantization: QuantizationMode,
}

impl UserDefinedEmbeddingModel {
    pub fn new(onnx_file: Vec<u8>, tokenizer_files: TokenizerFiles) -> Self {
        Self {
            onnx_file,
            tokenizer_files,
            quantization: QuantizationMode::None,
            pooling: None,
        }
    }
    
    pub fn with_quantization(mut self, quantization: QuantizationMode) -> Self {
        self.quantization = quantization;
        self
    }
    
    pub fn with_pooling(mut self, pooling: Pooling) -> Self {
        self.pooling = Some(pooling);
        self
    }
}

/// Rust representation of the TextEmbedding model
pub struct TextEmbedding {
    pub tokenizer: Tokenizer,
    pub(crate) pooling: Option<Pooling>,
    pub(crate) session: Session,
    pub(crate) need_token_type_ids: bool,
    pub(crate) quantization: QuantizationMode,
}
