use crate::config::SemanticConfig;
use crate::error::{CoreError, Result};
use serde::{Deserialize, Serialize};

pub struct OllamaClient {
    base_url: String,
    model: String,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    prompt: &'a str,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct TagsResponse {
    models: Option<Vec<ModelTag>>,
}

#[derive(Deserialize)]
struct ModelTag {
    name: String,
}

#[derive(Serialize)]
struct PullRequest<'a> {
    name: &'a str,
}

impl OllamaClient {
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
        }
    }

    /// Generate an embedding for the given text.
    /// Returns `None` if Ollama is unavailable (graceful degradation).
    pub async fn embed(&self, text: &str) -> Result<Option<Vec<f32>>> {
        let url = format!("{}/api/embeddings", self.base_url);
        let req = EmbeddingRequest {
            model: &self.model,
            prompt: text,
        };

        let response = match self.client.post(&url).json(&req).send().await {
            Ok(r) => r,
            Err(e) if e.is_connect() || e.is_timeout() => return Ok(None),
            Err(e) => return Err(CoreError::Embedding(format!("request failed: {e}"))),
        };

        if !response.status().is_success() {
            return Err(CoreError::Embedding(format!(
                "ollama returned status {}",
                response.status()
            )));
        }

        let body: EmbeddingResponse = response
            .json()
            .await
            .map_err(|e| CoreError::Embedding(format!("parse response: {e}")))?;

        Ok(Some(body.embedding))
    }

    /// Check if Ollama is reachable.
    pub async fn health_check(&self) -> bool {
        let url = format!("{}/api/tags", self.base_url);
        matches!(self.client.get(&url).send().await, Ok(r) if r.status().is_success())
    }

    /// Check if a specific model is available locally in Ollama.
    pub async fn has_model(&self, model: &str) -> bool {
        let url = format!("{}/api/tags", self.base_url);
        let response = match self.client.get(&url).send().await {
            Ok(r) if r.status().is_success() => r,
            _ => return false,
        };

        let tags: TagsResponse = match response.json().await {
            Ok(t) => t,
            Err(_) => return false,
        };

        tags.models
            .unwrap_or_default()
            .iter()
            .any(|m| m.name == model || m.name.starts_with(&format!("{model}:")))
    }

    /// Pull a model from the Ollama registry. Returns `true` if the pull succeeds.
    /// Uses a 10-minute timeout since model downloads can be large.
    pub async fn pull_model(&self, model: &str) -> Result<bool> {
        let url = format!("{}/api/pull", self.base_url);
        let pull_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()
            .unwrap_or_default();

        let response = match pull_client
            .post(&url)
            .json(&PullRequest { name: model })
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) if e.is_connect() || e.is_timeout() => return Ok(false),
            Err(e) => return Err(CoreError::Embedding(format!("pull request failed: {e}"))),
        };

        Ok(response.status().is_success())
    }

    /// List locally available model names from `GET {base}/api/tags`, sorted.
    /// Transport/non-2xx errors propagate so the UI can fall back to
    /// free-text model entry.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/api/tags", self.base_url);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| CoreError::Embedding(format!("list models failed: {e}")))?;
        if !response.status().is_success() {
            return Err(CoreError::Embedding(format!(
                "ollama returned status {}",
                response.status()
            )));
        }
        let tags: TagsResponse = response
            .json()
            .await
            .map_err(|e| CoreError::Embedding(format!("parse response: {e}")))?;
        let mut names: Vec<String> = tags
            .models
            .unwrap_or_default()
            .into_iter()
            .map(|m| m.name)
            .collect();
        names.sort();
        Ok(names)
    }

    /// Get the model name this client is configured for.
    pub fn model(&self) -> &str {
        &self.model
    }
}

/// OpenAI-compatible embedding response shape: `{"data": [{"embedding": [...]}]}`.
#[derive(Deserialize)]
struct OpenAiEmbeddingResponse {
    data: Vec<OpenAiEmbeddingData>,
}

#[derive(Deserialize)]
struct OpenAiEmbeddingData {
    embedding: Vec<f32>,
}

/// OpenAI-compatible embedding client (`POST {base}/embeddings`).
///
/// Covers OpenAI, OpenRouter, LM Studio, and self-hosted gateways with one
/// shape. Mirrors the `OllamaClient` graceful-degradation contract:
/// connect/timeout returns `Ok(None)` (caller keeps keyword-only index).
pub struct OpenAiEmbeddingClient {
    base_url: String,
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl OpenAiEmbeddingClient {
    pub fn new(base_url: &str, api_key: &str, model: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
        }
    }

    fn with_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if self.api_key.is_empty() {
            req
        } else {
            req.bearer_auth(&self.api_key)
        }
    }

    /// Generate an embedding for the given text.
    /// Returns `None` when the endpoint is unreachable (connect/timeout).
    pub async fn embed(&self, text: &str) -> Result<Option<Vec<f32>>> {
        let url = format!("{}/embeddings", self.base_url);
        let body = serde_json::json!({
            "model": self.model,
            "input": text,
        });

        let response = match self.with_auth(self.client.post(&url).json(&body)).send().await {
            Ok(r) => r,
            Err(e) if e.is_connect() || e.is_timeout() => return Ok(None),
            Err(e) => return Err(CoreError::Embedding(format!("request failed: {e}"))),
        };

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            let snippet: String = text.chars().take(500).collect();
            return Err(CoreError::Embedding(format!(
                "embedding API returned {status}: {snippet}"
            )));
        }

        let parsed: OpenAiEmbeddingResponse = response
            .json()
            .await
            .map_err(|e| CoreError::Embedding(format!("parse response: {e}")))?;
        match parsed.data.into_iter().next() {
            Some(d) => Ok(Some(d.embedding)),
            None => Err(CoreError::Embedding("embedding API returned no data".to_string())),
        }
    }

    /// Health check via `GET {base}/models` — never `POST /embeddings`,
    /// so a check does not bill an embedding call.
    pub async fn health_check(&self) -> bool {
        let url = format!("{}/models", self.base_url);
        matches!(
            self.with_auth(
                self.client
                    .get(&url)
                    .timeout(std::time::Duration::from_secs(15))
            )
            .send()
            .await,
            Ok(r) if r.status().is_success()
        )
    }

    /// Model ids from `GET {base}/models` (`data[].id`), sorted.
    /// Transport/non-2xx errors propagate so the UI can fall back to
    /// free-text model entry.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/models", self.base_url);
        let response = self
            .with_auth(
                self.client
                    .get(&url)
                    .timeout(std::time::Duration::from_secs(15)),
            )
            .send()
            .await
            .map_err(|e| CoreError::Embedding(format!("list models failed: {e}")))?;
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            let snippet: String = text.chars().take(500).collect();
            return Err(CoreError::Embedding(format!(
                "embedding API returned {status}: {snippet}"
            )));
        }
        let v: serde_json::Value = response
            .json()
            .await
            .map_err(|e| CoreError::Embedding(format!("parse models response: {e}")))?;
        let mut ids: Vec<String> = v
            .get("data")
            .and_then(|d| d.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        m.get("id").and_then(|id| id.as_str()).map(str::to_string)
                    })
                    .collect()
            })
            .unwrap_or_default();
        ids.sort();
        Ok(ids)
    }

    /// Get the model name this client is configured for.
    pub fn model(&self) -> &str {
        &self.model
    }
}

/// Unified embedding dispatch over the configured `SemanticConfig.provider`.
///
/// `"ollama"` (default) delegates to `OllamaClient`; `"api"` delegates to
/// `OpenAiEmbeddingClient`. Anything else falls back to Ollama.
pub enum EmbeddingClient {
    Ollama(OllamaClient),
    Api(OpenAiEmbeddingClient),
}

impl EmbeddingClient {
    /// Build the active backend from config. Missing `api_base_url`/`api_model`
    /// when `provider == "api"` is a construction error (the daemon logs a
    /// warning and skips embeddings for the run).
    pub fn from_config(c: &SemanticConfig) -> Result<Self> {
        if c.provider == "api" {
            if c.api_base_url.trim().is_empty() {
                return Err(CoreError::Embedding(
                    "semantic API provider selected but api_base_url is empty".to_string(),
                ));
            }
            if c.api_model.trim().is_empty() {
                return Err(CoreError::Embedding(
                    "semantic API provider selected but api_model is empty".to_string(),
                ));
            }
            Ok(Self::Api(OpenAiEmbeddingClient::new(
                &c.api_base_url,
                &c.api_key,
                &c.api_model,
            )))
        } else {
            Ok(Self::Ollama(OllamaClient::new(&c.ollama_url, &c.model)))
        }
    }

    /// Embed `text`, normalizing the vector to `dimensions` when nonzero:
    /// `0` accepts whatever the API returns; otherwise truncate or zero-pad.
    /// `vec0` rows and RRF fusion assume uniform length, so this normalize
    /// step is load-bearing for the API path (models vary in dimension).
    pub async fn embed(&self, text: &str, dimensions: usize) -> Result<Option<Vec<f32>>> {
        let raw = match self {
            Self::Ollama(c) => c.embed(text).await?,
            Self::Api(c) => c.embed(text).await?,
        };
        Ok(raw.map(|v| normalize_dimensions(v, dimensions)))
    }

    pub async fn health_check(&self) -> bool {
        match self {
            Self::Ollama(c) => c.health_check().await,
            Self::Api(c) => c.health_check().await,
        }
    }

    /// List available model ids for the Settings UI datalist.
    /// Ollama reports local tags; API reports `GET {base}/models` ids.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        match self {
            Self::Ollama(c) => c.list_models().await,
            Self::Api(c) => c.list_models().await,
        }
    }

    pub fn model(&self) -> &str {
        match self {
            Self::Ollama(c) => c.model(),
            Self::Api(c) => c.model(),
        }
    }

    pub fn is_api(&self) -> bool {
        matches!(self, Self::Api(_))
    }
}

/// Truncate or zero-pad `v` to exactly `dimensions` floats.
/// `0` (auto) returns `v` unchanged.
fn normalize_dimensions(mut v: Vec<f32>, dimensions: usize) -> Vec<f32> {
    if dimensions == 0 || v.len() == dimensions {
        return v;
    }
    if v.len() > dimensions {
        v.truncate(dimensions);
    } else {
        v.resize(dimensions, 0.0);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_config_defaults_to_ollama() {
        let c = SemanticConfig::default();
        let client = EmbeddingClient::from_config(&c).unwrap();
        assert!(!client.is_api());
        assert_eq!(client.model(), "nomic-embed-text");
    }

    #[test]
    fn from_config_api_requires_base_url_and_model() {
        let mut c = SemanticConfig::default();
        c.provider = "api".to_string();
        assert!(EmbeddingClient::from_config(&c).is_err());

        c.api_base_url = "https://api.openai.com/v1".to_string();
        assert!(EmbeddingClient::from_config(&c).is_err());

        c.api_model = "text-embedding-3-small".to_string();
        let client = EmbeddingClient::from_config(&c).unwrap();
        assert!(client.is_api());
        assert_eq!(client.model(), "text-embedding-3-small");
    }

    #[test]
    fn normalize_dimensions_pads_truncates_and_passes_through() {
        assert_eq!(normalize_dimensions(vec![1.0, 2.0], 0), vec![1.0, 2.0]);
        assert_eq!(normalize_dimensions(vec![1.0, 2.0], 2), vec![1.0, 2.0]);
        assert_eq!(normalize_dimensions(vec![1.0, 2.0, 3.0], 2), vec![1.0, 2.0]);
        assert_eq!(normalize_dimensions(vec![1.0], 3), vec![1.0, 0.0, 0.0]);
    }
}
