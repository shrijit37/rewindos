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
    /// so a check does not bill an embedding call. `Ok(false)` only on
    /// connect/timeout; `Err` carries HTTP failures for the Settings UI.
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/models", self.base_url);
        let response = match self
            .with_auth(
                self.client
                    .get(&url)
                    .timeout(std::time::Duration::from_secs(15)),
            )
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) if e.is_connect() || e.is_timeout() => return Ok(false),
            Err(e) => return Err(CoreError::Embedding(format!("health check failed: {e}"))),
        };
        if response.status().is_success() {
            Ok(true)
        } else {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            let snippet: String = text.chars().take(500).collect();
            Err(CoreError::Embedding(format!(
                "embedding API returned {status}: {snippet}"
            )))
        }
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

/// Gateway root: peel trailing endpoint segments so a full endpoint URL
/// (`…/v1`, `…/api/embeddings`, `…/v1/ocr`, …) still resolves to the host root.
/// Only path segments are stripped — never the host itself.
pub(crate) fn gateway_root(base: &str) -> String {
    let mut s = base.trim_end_matches('/').to_string();
    loop {
        let stripped = ["v1", "embeddings", "embed", "api", "ocr"]
            .iter()
            .find_map(|seg| s.strip_suffix(&format!("/{seg}")).map(str::to_string));
        match stripped {
            Some(root) => s = root.trim_end_matches('/').to_string(),
            None => return s,
        }
    }
}

 /// Custom gateway embedding client (self-hosted semantic-OCR gateway).
///
/// Vectors come from the gateway's OpenAI-shaped `POST {root}/v1/embeddings`;
/// checks use its unauthenticated `GET {root}/health` — the gateway exposes
/// no `/models` endpoint, so the OpenAI client's probe 404s there.
/// Graceful degradation matches `OllamaClient`: connect/timeout → `Ok(None)`.
pub struct GatewayEmbeddingClient {
    root: String,
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl GatewayEmbeddingClient {
    pub fn new(base_url: &str, api_key: &str, model: &str) -> Self {
        Self {
            root: gateway_root(base_url),
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
    /// Returns `None` when the gateway is unreachable (connect/timeout).
    pub async fn embed(&self, text: &str) -> Result<Option<Vec<f32>>> {
        // Gateway rejects inputs over 4000 chars (413) — truncate by chars.
        // A prefix embedding keeps the screenshot searchable; a 413 would
        // drop it from semantic search entirely.
        let input: String = text.chars().take(4000).collect();
        let url = format!("{}/v1/embeddings", self.root);
        let body = serde_json::json!({
            "model": self.model,
            "input": input,
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
                "embedding gateway returned {status}: {snippet}"
            )));
        }
        let parsed: OpenAiEmbeddingResponse = response
            .json()
            .await
            .map_err(|e| CoreError::Embedding(format!("parse response: {e}")))?;
        match parsed.data.into_iter().next() {
            Some(d) => Ok(Some(d.embedding)),
            None => Err(CoreError::Embedding("embedding gateway returned no data".to_string())),
        }
    }

    /// `GET {root}/health` needs no auth. `Ok(false)` only on
    /// connect/timeout; `Err` carries HTTP failures for the Settings UI.
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/health", self.root);
        let response = match self
            .with_auth(
                self.client
                    .get(&url)
                    .timeout(std::time::Duration::from_secs(15)),
            )
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) if e.is_connect() || e.is_timeout() => return Ok(false),
            Err(e) => return Err(CoreError::Embedding(format!("health check failed: {e}"))),
        };
        if response.status().is_success() {
            Ok(true)
        } else {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            let snippet: String = text.chars().take(500).collect();
            Err(CoreError::Embedding(format!(
                "gateway health returned {status}: {snippet}"
            )))
        }
    }

    /// The gateway runs one fixed embedding model — report the configured one
    /// so the Settings datalist still offers something.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![self.model.clone()])
    }

    /// Get the model name this client is configured for.
    pub fn model(&self) -> &str {
        &self.model
    }
}

/// Unified embedding dispatch over the configured `SemanticConfig.provider`.
///
/// `"ollama"` (default) delegates to `OllamaClient`; `"api"` delegates to
/// `OpenAiEmbeddingClient`; `"gateway"` delegates to `GatewayEmbeddingClient`
/// (custom semantic-OCR gateway: `POST {root}/v1/embeddings`,
/// `GET {root}/health`). Anything else falls back to Ollama.
pub enum EmbeddingClient {
    Ollama(OllamaClient),
    Api(OpenAiEmbeddingClient),
    Gateway(GatewayEmbeddingClient),
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
        } else if c.provider == "gateway" {
            if c.api_base_url.trim().is_empty() {
                return Err(CoreError::Embedding(
                    "gateway provider selected but api_base_url is empty".to_string(),
                ));
            }
            if c.api_model.trim().is_empty() {
                return Err(CoreError::Embedding(
                    "gateway provider selected but api_model is empty".to_string(),
                ));
            }
            Ok(Self::Gateway(GatewayEmbeddingClient::new(
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
            Self::Gateway(c) => c.embed(text).await?,
        };
        Ok(raw.map(|v| normalize_dimensions(v, dimensions)))
    }

    /// `Ok(true)` reachable, `Ok(false)` only on connect/timeout, `Err`
    /// carries HTTP failures so the Settings UI can say why a check failed.
    pub async fn health_check(&self) -> Result<bool> {
        match self {
            Self::Ollama(c) => Ok(c.health_check().await),
            Self::Api(c) => c.health_check().await,
            Self::Gateway(c) => c.health_check().await,
        }
    }

    /// List available model ids for the Settings UI datalist.
    /// Ollama reports local tags; API reports `GET {base}/models` ids;
    /// gateway reports the configured model (it runs one fixed model).
    pub async fn list_models(&self) -> Result<Vec<String>> {
        match self {
            Self::Ollama(c) => c.list_models().await,
            Self::Api(c) => c.list_models().await,
            Self::Gateway(c) => c.list_models().await,
        }
    }

    pub fn model(&self) -> &str {
        match self {
            Self::Ollama(c) => c.model(),
            Self::Api(c) => c.model(),
            Self::Gateway(c) => c.model(),
        }
    }

    /// True for any remote backend (`Api` or `Gateway`); false for local Ollama.
    pub fn is_api(&self) -> bool {
        !matches!(self, Self::Ollama(_))
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

    #[test]
    fn gateway_root_strips_v1_and_slashes() {
        assert_eq!(gateway_root("https://host/v1"), "https://host");
        assert_eq!(gateway_root("https://host/v1/"), "https://host");
        assert_eq!(gateway_root("https://host"), "https://host");
        assert_eq!(gateway_root("https://host/"), "https://host");
        assert_eq!(gateway_root("https://host/api/embeddings"), "https://host");
        assert_eq!(gateway_root("https://host/v1/embeddings"), "https://host");
        assert_eq!(gateway_root("https://host/v1/ocr"), "https://host");
        assert_eq!(gateway_root("https://host/subpath/v1"), "https://host/subpath");
    }

    #[test]
    fn from_config_gateway_requires_base_url_and_model() {
        let mut c = SemanticConfig::default();
        c.provider = "gateway".to_string();
        assert!(EmbeddingClient::from_config(&c).is_err());
        c.api_base_url = "https://gateway.local".to_string();
        assert!(EmbeddingClient::from_config(&c).is_err());
        c.api_model = "nomic-embed-text".to_string();
        let client = EmbeddingClient::from_config(&c).unwrap();
        assert!(client.is_api());
        assert_eq!(client.model(), "nomic-embed-text");
    }

    /// Route-aware stub: `GET /health` → ok, `POST /v1/embeddings` →
    /// OpenAI-shaped vector. Anything else 404s (like the real gateway's
    /// missing `/v1/models`).
    async fn spawn_gateway_stub() -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else { break };
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    let mut req = Vec::new();
                    let mut buf = vec![0u8; 8192];
                    let mut content_len = 0usize;
                    let mut header_len = None::<usize>;
                    while header_len.map(|h| req.len() < h + content_len).unwrap_or(true) {
                        match socket.read(&mut buf).await {
                            Ok(0) => break,
                            Ok(n) => req.extend_from_slice(&buf[..n]),
                            Err(_) => break,
                        };
                        if header_len.is_none() {
                            if let Some(p) = find_stub_header_end(&req) {
                                header_len = Some(p);
                                content_len = parse_stub_content_length(&req[..p]);
                            }
                        }
                        if req.len() > 8_000_000 {
                            break;
                        }
                    }
                    let head = String::from_utf8_lossy(&req);
                    let first_line = head.lines().next().unwrap_or("");
                    let (status, body) = if first_line.starts_with("GET /health") {
                        (200, "{\"status\":\"ok\"}".to_string())
                    } else if first_line.starts_with("POST /v1/embeddings") {
                        (200, "{\"object\":\"list\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[0.1,0.2,0.3]}]}".to_string())
                    } else {
                        (404, "{\"detail\":\"Not Found\"}".to_string())
                    };
                    let resp = format!(
                        "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = socket.write_all(resp.as_bytes()).await;
                });
            }
        });
        (format!("http://{addr}"), handle)
    }

    fn find_stub_header_end(req: &[u8]) -> Option<usize> {
        req.windows(4)
            .position(|w| w == b"\r\n\r\n")
            .map(|p| p + 4)
    }

    fn parse_stub_content_length(head: &[u8]) -> usize {
        String::from_utf8_lossy(head)
            .lines()
            .skip(1)
            .filter_map(|l| {
                let (k, v) = l.split_once(':')?;
                (k.trim().eq_ignore_ascii_case("content-length"))
                    .then(|| v.trim().parse().unwrap_or(0))
            })
            .next()
            .unwrap_or(0)
    }

    #[tokio::test]
    async fn gateway_embed_and_health_hit_stub() {
        let (root, handle) = spawn_gateway_stub().await;
        let c = GatewayEmbeddingClient::new(&root, "k", "m");
        assert!(c.health_check().await.unwrap());
        assert_eq!(c.embed("hi").await.unwrap(), Some(vec![0.1, 0.2, 0.3]));
        // A `/v1`-suffixed base resolves to the same root.
        let c2 = GatewayEmbeddingClient::new(&format!("{root}/v1"), "", "m");
        assert!(c2.health_check().await.unwrap());
        // The OpenAI-shaped probe 404s here — the failure this client avoids.
        let openai = OpenAiEmbeddingClient::new(&format!("{root}/v1"), "", "m");
        assert!(!openai.health_check().await.unwrap_or(false));
        handle.abort();
    }
}
