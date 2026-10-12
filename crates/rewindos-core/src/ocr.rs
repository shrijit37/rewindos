use std::path::Path;
use std::process::Output;
use std::sync::Arc;
use std::time::Duration;

use tokio::process::Command;
use tracing::{debug, warn};

use crate::config::OcrConfig;
use crate::error::{CoreError, Result};
use crate::paddle_ocr::PaddleOcrSidecar;
use crate::schema::NewBoundingBox;

/// Result of running OCR on a single screenshot.
pub struct OcrOutput {
    pub full_text: String,
    pub bounding_boxes: Vec<NewBoundingBox>,
    pub word_count: i32,
}

/// Which OCR engine to use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OcrEngine {
    Tesseract,
    PaddleOcr,
    Api,
    Gateway,
}

impl OcrEngine {
    pub fn from_config(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "paddleocr" | "paddle" => Self::PaddleOcr,
            "api" => Self::Api,
            "gateway" => Self::Gateway,
            _ => Self::Tesseract,
        }
    }
}

/// Unified OCR dispatcher — runs the configured engine.
pub async fn run_ocr(
    engine: &OcrEngine,
    image_path: &Path,
    lang: &str,
    paddle_sidecar: Option<&Arc<PaddleOcrSidecar>>,
    api_client: Option<&ApiOcrClient>,
    gateway_client: Option<&GatewayOcrClient>,
) -> Result<OcrOutput> {
    match engine {
        OcrEngine::Tesseract => run_tesseract(image_path, lang).await,
        OcrEngine::PaddleOcr => {
            let sidecar = paddle_sidecar.ok_or_else(|| {
                CoreError::Ocr("PaddleOCR sidecar not available".to_string())
            })?;
            sidecar.run_ocr(image_path).await
        }
        OcrEngine::Api => {
            let client = api_client.ok_or_else(|| {
                CoreError::Ocr("API OCR client not configured".to_string())
            })?;
            client.run_api_ocr(image_path).await
        }
        OcrEngine::Gateway => {
            let client = gateway_client.ok_or_else(|| {
                CoreError::Ocr("gateway OCR client not configured".to_string())
            })?;
            client.run_gateway_ocr(image_path).await
        }
    }
}
/// Minimum confidence threshold for including a word (0-100).
const MIN_CONFIDENCE: f32 = 30.0;

/// Maximum time to wait for a tesseract process before killing it.
const TESSERACT_TIMEOUT: Duration = Duration::from_secs(10);

/// Run Tesseract OCR on an image file, returning extracted text and bounding boxes.
///
/// Spawns `tesseract` CLI as a subprocess with TSV output format.
/// Filters out low-confidence words (below 30%).
/// Times out after 10 seconds.
pub async fn run_tesseract(image_path: &Path, lang: &str) -> Result<OcrOutput> {
    let output = spawn_tesseract(image_path, lang).await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(CoreError::Ocr(format!(
            "tesseract exited with {}: {stderr}",
            output.status
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_tsv_output(&stdout)
}

async fn spawn_tesseract(image_path: &Path, lang: &str) -> Result<Output> {
    let child = Command::new("tesseract")
        .arg(image_path)
        .arg("stdout")
        .args(["--oem", "1"])
        .args(["--psm", "3"])
        .args(["-l", lang])
        .arg("tsv")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        // A timed-out OCR drops `wait_with_output` but not the child, so the
        // tesseract process keeps running to completion unconsumed. Repeated
        // timeouts then stack up and saturate every core.
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| CoreError::Ocr(format!("failed to spawn tesseract: {e}")))?;

    let result = tokio::time::timeout(TESSERACT_TIMEOUT, child.wait_with_output()).await;

    match result {
        Ok(Ok(output)) => Ok(output),
        Ok(Err(e)) => Err(CoreError::Ocr(format!("tesseract process error: {e}"))),
        Err(_) => Err(CoreError::Ocr("tesseract timed out after 10s".to_string())),
    }
}

/// Parse Tesseract TSV output into full text and bounding boxes.
///
/// TSV format (level 5 = individual words):
/// ```text
/// level  page_num  block_num  par_num  line_num  word_num  left  top  width  height  conf  text
/// ```
fn parse_tsv_output(tsv: &str) -> Result<OcrOutput> {
    let mut full_text_parts: Vec<String> = Vec::new();
    let mut bounding_boxes = Vec::new();
    let mut current_line_num: i32 = -1;
    let mut word_count: i32 = 0;

    for line in tsv.lines().skip(1) {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 12 {
            continue;
        }

        let level: i32 = match fields[0].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Level 5 = individual words
        if level != 5 {
            continue;
        }

        let confidence: f32 = fields[10].parse().unwrap_or(-1.0);
        let text = fields[11].trim();

        if text.is_empty() || confidence < MIN_CONFIDENCE {
            continue;
        }

        let line_num: i32 = fields[4].parse().unwrap_or(0);

        // Insert newline when we move to a new line
        if current_line_num >= 0 && line_num != current_line_num && !full_text_parts.is_empty() {
            full_text_parts.push("\n".to_string());
        }
        current_line_num = line_num;

        full_text_parts.push(text.to_string());
        word_count += 1;

        let left: i32 = fields[6].parse().unwrap_or(0);
        let top: i32 = fields[7].parse().unwrap_or(0);
        let width: i32 = fields[8].parse().unwrap_or(0);
        let height: i32 = fields[9].parse().unwrap_or(0);

        bounding_boxes.push(NewBoundingBox {
            text_content: text.to_string(),
            x: left,
            y: top,
            width,
            height,
            confidence: Some(confidence as f64),
        });
    }

    let full_text = join_words(&full_text_parts);

    debug!(
        words = word_count,
        boxes = bounding_boxes.len(),
        "parsed tesseract output"
    );

    Ok(OcrOutput {
        full_text,
        bounding_boxes,
        word_count,
    })
}

/// Join word parts with spaces, collapsing around newlines.
fn join_words(parts: &[String]) -> String {
    let mut result = String::new();
    for (i, part) in parts.iter().enumerate() {
        if part == "\n" {
            result.push('\n');
        } else {
            // Add space before word unless at start or after newline
            if i > 0 && !result.is_empty() && !result.ends_with('\n') {
                result.push(' ');
            }
            result.push_str(part);
        }
    }
    result
}

/// Check if tesseract is available on the system.
pub async fn is_tesseract_available() -> bool {
    match Command::new("tesseract").arg("--version").output().await {
        Ok(output) => {
            if output.status.success() {
                let version = String::from_utf8_lossy(&output.stdout);
                debug!(version = %version.lines().next().unwrap_or("unknown"), "tesseract found");
                true
            } else {
                warn!("tesseract found but returned non-zero status");
                false
            }
        }
        Err(e) => {
            warn!(error = %e, "tesseract not found");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_TSV: &str = "\
level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext
1\t1\t0\t0\t0\t0\t0\t0\t1920\t1080\t-1\t
2\t1\t1\t0\t0\t0\t100\t50\t500\t200\t-1\t
3\t1\t1\t1\t0\t0\t100\t50\t500\t100\t-1\t
4\t1\t1\t1\t1\t0\t100\t50\t500\t25\t-1\t
5\t1\t1\t1\t1\t1\t100\t50\t80\t20\t96.5\tHello
5\t1\t1\t1\t1\t2\t190\t50\t90\t20\t94.2\tWorld
4\t1\t1\t1\t2\t0\t100\t80\t500\t25\t-1\t
5\t1\t1\t1\t2\t1\t100\t80\t120\t20\t91.0\tSecond
5\t1\t1\t1\t2\t2\t230\t80\t60\t20\t88.3\tLine";

    const TSV_WITH_LOW_CONFIDENCE: &str = "\
level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext
5\t1\t1\t1\t1\t1\t100\t50\t80\t20\t95.0\tGood
5\t1\t1\t1\t1\t2\t190\t50\t90\t20\t10.0\tNoisy
5\t1\t1\t1\t1\t3\t290\t50\t70\t20\t85.0\tWord";

    #[test]
    fn parse_tsv_should_extract_text_and_boxes() {
        let result = parse_tsv_output(SAMPLE_TSV).unwrap();

        assert_eq!(result.full_text, "Hello World\nSecond Line");
        assert_eq!(result.word_count, 4);
        assert_eq!(result.bounding_boxes.len(), 4);
    }

    #[test]
    fn parse_tsv_should_populate_bounding_box_coordinates() {
        let result = parse_tsv_output(SAMPLE_TSV).unwrap();

        let first_box = &result.bounding_boxes[0];
        assert_eq!(first_box.text_content, "Hello");
        assert_eq!(first_box.x, 100);
        assert_eq!(first_box.y, 50);
        assert_eq!(first_box.width, 80);
        assert_eq!(first_box.height, 20);
        assert!(
            (first_box.confidence.unwrap() - 96.5).abs() < 0.1,
            "confidence should be ~96.5"
        );
    }

    #[test]
    fn parse_tsv_should_filter_low_confidence_words() {
        let result = parse_tsv_output(TSV_WITH_LOW_CONFIDENCE).unwrap();

        assert_eq!(result.full_text, "Good Word");
        assert_eq!(result.word_count, 2);
        assert_eq!(result.bounding_boxes.len(), 2);
    }

    #[test]
    fn parse_tsv_should_handle_empty_input() {
        let tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n";
        let result = parse_tsv_output(tsv).unwrap();

        assert!(result.full_text.is_empty());
        assert_eq!(result.word_count, 0);
        assert!(result.bounding_boxes.is_empty());
    }

    #[test]
    fn join_words_should_space_separate_on_same_line() {
        let parts = vec!["Hello".to_string(), "World".to_string()];
        assert_eq!(join_words(&parts), "Hello World");
    }

    #[test]
    fn join_words_should_newline_between_lines() {
        let parts = vec!["Line1".to_string(), "\n".to_string(), "Line2".to_string()];
        assert_eq!(join_words(&parts), "Line1\nLine2");
    }
}

// -- API OCR (OpenAI-compatible vision) --

/// Vision OCR client: uploads the screenshot to `POST {base}/chat/completions`
/// as an `image_url` part and returns the transcription.
///
/// Providers return text only — no geometry — so `bounding_boxes` is always
/// empty in API mode. Keyword/semantic search works unchanged; word-level
/// overlays show nothing.
pub struct ApiOcrClient {
    base_url: String,
    api_key: String,
    model: String,
    timeout_secs: u64,
    max_width: u32,
    client: reqwest::Client,
}

impl ApiOcrClient {
    /// Validate the required fields (`api_base_url`, `api_model`); key is
    /// optional (empty = no `Authorization` header, for self-hosted gateways).
    pub fn from_config(c: &OcrConfig) -> Result<Self> {
        if c.api_base_url.trim().is_empty() {
            return Err(CoreError::Ocr("API OCR selected but api_base_url is empty".to_string()));
        }
        if c.api_model.trim().is_empty() {
            return Err(CoreError::Ocr("API OCR selected but api_model is empty".to_string()));
        }
        Ok(Self {
            base_url: c.api_base_url.trim_end_matches('/').to_string(),
            api_key: c.api_key.clone(),
            model: c.api_model.clone(),
            timeout_secs: c.api_timeout_secs.max(1),
            max_width: c.api_max_width.max(1),
            // No global timeout (long generations); each request sets its own.
            client: reqwest::Client::builder()
                .connect_timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
        })
    }

    /// Health check via `GET {base}/models`. Unreachable endpoint returns
    /// `Ok(false)` only on connect/timeout (never throws there — matches
    /// `ChatClient` semantics); `Err` carries HTTP failures for the UI.
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/models", self.base_url);
        let mut req = self
            .client
            .get(&url)
            .timeout(std::time::Duration::from_secs(15));
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let response = match req.send().await {
            Ok(r) => r,
            Err(e) if e.is_connect() || e.is_timeout() => return Ok(false),
            Err(e) => return Err(CoreError::Ocr(format!("health check failed: {e}"))),
        };
        if response.status().is_success() {
            Ok(true)
        } else {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            let snippet: String = text.chars().take(500).collect();
            Err(CoreError::Ocr(format!(
                "provider health returned {status}: {snippet}"
            )))
        }
    }

    /// Model ids from `GET {base}/models` (`data[].id`), sorted.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/models", self.base_url);
        let mut req = self
            .client
            .get(&url)
            .timeout(std::time::Duration::from_secs(15));
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let response = req
            .send()
            .await
            .map_err(|e| CoreError::Ocr(format!("list models failed: {e}")))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let snippet: String = body.chars().take(500).collect();
            return Err(CoreError::Ocr(format!("provider returned {status}: {snippet}")));
        }
        let v: serde_json::Value = response
            .json()
            .await
            .map_err(|e| CoreError::Ocr(format!("parse models response: {e}")))?;
        let mut ids: Vec<String> = v
            .get("data")
            .and_then(|d| d.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m.get("id").and_then(|id| id.as_str()).map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        ids.sort();
        Ok(ids)
    }

    /// Transcribe `image_path` via the vision endpoint.
    ///
    /// Unlike embeddings there is no skip-and-continue semantic: every
    /// failure (including connect/timeout) is `Err`, and the caller marks
    /// the screenshot `OcrStatus::Failed` — same as tesseract failure today.
    pub async fn run_api_ocr(&self, image_path: &Path) -> Result<OcrOutput> {
        let bytes = self.upload_bytes(image_path)?;
        let data_url = format!("data:image/webp;base64,{}", encode_base64(&bytes));
        let url = format!("{}/chat/completions", self.base_url);
        let body = serde_json::json!({
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe all visible text exactly, preserving reading order. Return only the transcription."},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }],
            "max_tokens": 4096,
            "temperature": 0
        });

        let mut req = self
            .client
            .post(&url)
            .json(&body)
            .timeout(std::time::Duration::from_secs(self.timeout_secs));
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let response = req
            .send()
            .await
            .map_err(|e| CoreError::Ocr(format!("API OCR request failed: {e}")))?;
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            let snippet: String = text.chars().take(500).collect();
            return Err(CoreError::Ocr(format!("provider returned {status}: {snippet}")));
        }
        let v: serde_json::Value = response
            .json()
            .await
            .map_err(|e| CoreError::Ocr(format!("parse OCR response: {e}")))?;
        let full_text = parse_chat_content(&v)?;
        Ok(OcrOutput {
            word_count: full_text.split_whitespace().count() as i32,
            full_text,
            bounding_boxes: vec![],
        })
    }

    /// Raw file bytes when the image fits, else a downscaled WebP re-encode.
    /// Screenshots on disk are already WebP, so the fast path uploads as-is.
    fn upload_bytes(&self, image_path: &Path) -> Result<Vec<u8>> {
        read_upload_bytes(image_path, self.max_width)
    }
}

/// Raw file bytes when the image fits `max_width`, else a downscaled WebP
/// re-encode. Shared by the OpenAI-vision and gateway upload paths.
fn read_upload_bytes(image_path: &Path, max_width: u32) -> Result<Vec<u8>> {
    let (width, _) = image::image_dimensions(image_path)
        .map_err(|e| CoreError::Ocr(format!("cannot read image dimensions: {e}")))?;
    if width <= max_width {
        return Ok(std::fs::read(image_path)?);
    }
    let img = image::open(image_path)
        .map_err(|e| CoreError::Ocr(format!("cannot decode image: {e}")))?;
    let small = crate::hasher::downscale_for_storage(&img, max_width);
    let encoded = webp::Encoder::from_image(&small)
        .map_err(|e| CoreError::Ocr(format!("webp re-encode failed: {e}")))?;
    Ok(encoded.encode(80.0).to_vec())
}

// -- Gateway OCR (custom semantic-OCR gateway, multipart POST {root}/ocr) --

/// OCR client for a self-hosted semantic-OCR gateway (`POST {root}/ocr`,
/// multipart `file` field). Unlike the OpenAI-vision path the gateway
/// returns word/line geometry as quads, so `bounding_boxes` is populated.
pub struct GatewayOcrClient {
    root: String,
    api_key: String,
    timeout_secs: u64,
    max_width: u32,
    client: reqwest::Client,
}

impl GatewayOcrClient {
    /// Only `api_base_url` is required (gateway root, `/v1` suffix tolerated);
    /// key is optional at construction (the server 401s without one, surfaced
    /// per screenshot). No model: the gateway runs one fixed OCR engine.
    pub fn from_config(c: &OcrConfig) -> Result<Self> {
        if c.api_base_url.trim().is_empty() {
            return Err(CoreError::Ocr("gateway OCR selected but api_base_url is empty".to_string()));
        }
        Ok(Self {
            root: gateway_ocr_root(&c.api_base_url),
            api_key: c.api_key.clone(),
            timeout_secs: c.api_timeout_secs.max(1),
            max_width: c.api_max_width.max(1),
            client: reqwest::Client::builder()
                .connect_timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
        })
    }

    /// `GET {root}/health` (unauthenticated on the gateway). `Ok(false)` only
    /// on connect/timeout; `Err` carries HTTP failures for the Settings UI.
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/health", self.root);
        let mut req = self
            .client
            .get(&url)
            .timeout(std::time::Duration::from_secs(15));
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let response = match req.send().await {
            Ok(r) => r,
            Err(e) if e.is_connect() || e.is_timeout() => return Ok(false),
            Err(e) => return Err(CoreError::Ocr(format!("health check failed: {e}"))),
        };
        if response.status().is_success() {
            Ok(true)
        } else {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            let snippet: String = text.chars().take(500).collect();
            Err(CoreError::Ocr(format!(
                "gateway health returned {status}: {snippet}"
            )))
        }
    }

    /// The gateway runs one fixed OCR engine — nothing to pick.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![])
    }

    /// Transcribe `image_path` via `POST {root}/ocr` (multipart `file`).
    /// Failures are `Err` → caller marks `OcrStatus::Failed`, same as the
    /// other engines.
    pub async fn run_gateway_ocr(&self, image_path: &Path) -> Result<OcrOutput> {
        let bytes = self.upload_bytes(image_path)?;
        let (content_type, body) = encode_multipart_file(&bytes);
        let url = format!("{}/ocr", self.root);
        let mut req = self
            .client
            .post(&url)
            .header("Content-Type", content_type)
            .body(body)
            .timeout(std::time::Duration::from_secs(self.timeout_secs));
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let response = req
            .send()
            .await
            .map_err(|e| CoreError::Ocr(format!("gateway OCR request failed: {e}")))?;
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            let snippet: String = text.chars().take(500).collect();
            return Err(CoreError::Ocr(format!("gateway returned {status}: {snippet}")));
        }
        let v: serde_json::Value = response
            .json()
            .await
            .map_err(|e| CoreError::Ocr(format!("parse gateway OCR response: {e}")))?;
        parse_gateway_response(&v)
    }

    fn upload_bytes(&self, image_path: &Path) -> Result<Vec<u8>> {
        read_upload_bytes(image_path, self.max_width)
    }
}

/// Gateway root: trim slashes and a trailing `/v1` so both `https://host`
/// and `https://host/v1` resolve to the same root.
fn gateway_ocr_root(base: &str) -> String {
    let s = base.trim_end_matches('/').to_string();
    match s.strip_suffix("/v1") {
        Some(root) => root.trim_end_matches('/').to_string(),
        None => s,
    }
}

/// Minimal `multipart/form-data` encoder for a single `file` field.
/// Hand-rolled so the gateway path adds no dependency (reqwest `multipart`
/// is not enabled in this tree).
fn encode_multipart_file(bytes: &[u8]) -> (String, Vec<u8>) {
    let boundary = format!("rewindos{:016x}", bytes.len().wrapping_mul(0x9E3779B97F4A7C15));
    let mut body = Vec::with_capacity(bytes.len() + 256);
    body.extend_from_slice(b"--");
    body.extend_from_slice(boundary.as_bytes());
    body.extend_from_slice(
        b"\r\nContent-Disposition: form-data; name=\"file\"; filename=\"screenshot.webp\"\r\nContent-Type: image/webp\r\n\r\n",
    );
    body.extend_from_slice(bytes);
    body.extend_from_slice(b"\r\n--");
    body.extend_from_slice(boundary.as_bytes());
    body.extend_from_slice(b"--\r\n");
    (
        format!("multipart/form-data; boundary={boundary}"),
        body,
    )
}

/// Parse a gateway `/ocr` response:
/// `{"text": str, "confidence": f64|null, "boxes": [[[x,y] x4], ...]}`.
/// Quads become axis-aligned boxes; when the newline-split line count equals
/// the box count each box gets its line as `text_content`, else empty text
/// (geometry without word mapping — honest, overlays still render).
fn parse_gateway_response(v: &serde_json::Value) -> Result<OcrOutput> {
    let full_text = v
        .get("text")
        .and_then(|t| t.as_str())
        .ok_or_else(|| CoreError::Ocr("gateway OCR response has no text field".to_string()))?
        .to_string();
    let confidence = v.get("confidence").and_then(|c| c.as_f64());
    let quads = v
        .get("boxes")
        .and_then(|b| b.as_array())
        .cloned()
        .unwrap_or_default();
    let lines: Vec<&str> = full_text.lines().collect();
    let zip_text = lines.len() == quads.len();
    let mut bounding_boxes = Vec::with_capacity(quads.len());
    for (i, quad) in quads.iter().enumerate() {
        if let Some((x, y, width, height)) = quad_to_aabb(quad) {
            bounding_boxes.push(NewBoundingBox {
                text_content: if zip_text { lines[i].to_string() } else { String::new() },
                x,
                y,
                width,
                height,
                confidence,
            });
        }
    }
    Ok(OcrOutput {
        word_count: full_text.split_whitespace().count() as i32,
        full_text,
        bounding_boxes,
    })
}

/// Quad `[[x,y] x4]` (ints or floats) → axis-aligned `(x, y, w, h)`.
/// `None` for degenerate (zero-area) quads.
fn quad_to_aabb(quad: &serde_json::Value) -> Option<(i32, i32, i32, i32)> {
    let pts = quad.as_array()?;
    if pts.len() != 4 {
        return None;
    }
    let mut xs = Vec::with_capacity(4);
    let mut ys = Vec::with_capacity(4);
    for p in pts {
        let pair = p.as_array()?;
        if pair.len() != 2 {
            return None;
        }
        xs.push(pair[0].as_f64()?);
        ys.push(pair[1].as_f64()?);
    }
    let (min_x, max_x) = (xs.iter().cloned().fold(f64::INFINITY, f64::min), xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    let (min_y, max_y) = (ys.iter().cloned().fold(f64::INFINITY, f64::min), ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    let (x, y) = (min_x.round() as i32, min_y.round() as i32);
    let (width, height) = ((max_x - min_x).round() as i32, (max_y - min_y).round() as i32);
    if width <= 0 || height <= 0 {
        return None;
    }
    Some((x, y, width, height))
}

/// Extract the assistant text from a `POST /chat/completions` response.
/// Handles string content and array content-parts (text parts joined).
/// Present-but-empty content is `Ok("")` (blank screenshot, like tesseract
/// on empty input); a missing `choices[0].message.content` is `Err`.
fn parse_chat_content(v: &serde_json::Value) -> Result<String> {
    let content = v
        .pointer("/choices/0/message/content")
        .ok_or_else(|| CoreError::Ocr("API OCR response has no choices[0].message.content".to_string()))?;
    if let Some(s) = content.as_str() {
        return Ok(s.to_string());
    }
    if let Some(parts) = content.as_array() {
        let mut out = String::new();
        for part in parts {
            if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                out.push_str(t);
            }
        }
        return Ok(out);
    }
    Err(CoreError::Ocr("API OCR response content is not text".to_string()))
}

/// Standard base64 (RFC 4648 §4). Hand-rolled so API mode adds no new
/// dependency; covered by vector tests below.
fn encode_base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[((n >> 6) & 63) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[(n & 63) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

#[cfg(test)]
mod api_tests {
    use super::*;

    #[test]
    fn engine_from_config_maps_api() {
        assert_eq!(OcrEngine::from_config("api"), OcrEngine::Api);
        assert_eq!(OcrEngine::from_config("API"), OcrEngine::Api);
        assert_eq!(OcrEngine::from_config("paddleocr"), OcrEngine::PaddleOcr);
        assert_eq!(OcrEngine::from_config("paddle"), OcrEngine::PaddleOcr);
        assert_eq!(OcrEngine::from_config("tesseract"), OcrEngine::Tesseract);
        assert_eq!(OcrEngine::from_config("gateway"), OcrEngine::Gateway);
        assert_eq!(OcrEngine::from_config("anything-else"), OcrEngine::Tesseract);
    }

    #[test]
    fn api_client_requires_base_url_and_model() {
        let mut c = OcrConfig::default();
        assert!(ApiOcrClient::from_config(&c).is_err());
        c.api_base_url = "https://api.openai.com/v1/".to_string();
        assert!(ApiOcrClient::from_config(&c).is_err());
        c.api_model = "gpt-4o-mini".to_string();
        assert!(ApiOcrClient::from_config(&c).is_ok());
    }

    #[test]
    fn base64_matches_rfc_vectors() {
        assert_eq!(encode_base64(b""), "");
        assert_eq!(encode_base64(b"f"), "Zg==");
        assert_eq!(encode_base64(b"fo"), "Zm8=");
        assert_eq!(encode_base64(b"foo"), "Zm9v");
        assert_eq!(encode_base64(b"hello world"), "aGVsbG8gd29ybGQ=");
    }

    #[test]
    fn parse_chat_content_handles_shapes() {
        let s = serde_json::json!({"choices": [{"message": {"content": "hello"}}]});
        assert_eq!(parse_chat_content(&s).unwrap(), "hello");
        let arr = serde_json::json!({"choices": [{"message": {"content": [
            {"type": "text", "text": "hello "},
            {"type": "text", "text": "world"}
        ]}}]});
        assert_eq!(parse_chat_content(&arr).unwrap(), "hello world");
        let empty = serde_json::json!({"choices": [{"message": {"content": ""}}]});
        assert_eq!(parse_chat_content(&empty).unwrap(), "");
        let missing = serde_json::json!({"choices": []});
        assert!(parse_chat_content(&missing).is_err());
    }

    /// Minimal route-aware HTTP stub: `POST /chat/completions` returns a
    /// transcription, `GET /models` returns two ids. `Connection: close`
    /// per response so no keep-alive framing is needed.
    async fn spawn_stub() -> (String, tokio::task::JoinHandle<()>) {
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
                            if let Some(p) = find_header_end(&req) {
                                header_len = Some(p);
                                content_len = parse_content_length(&req[..p]);
                            }
                        }
                        if req.len() > 8_000_000 {
                            break;
                        }
                    }
                    let head = String::from_utf8_lossy(&req);
                    let first_line = head.lines().next().unwrap_or("");
                    let body = if first_line.starts_with("POST") {
                        "{\"choices\":[{\"message\":{\"content\":\"hello world\"}}]}"
                    } else {
                        "{\"data\":[{\"id\":\"gpt-4o-mini\"},{\"id\":\"gpt-4o\"}]}"
                    };
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = socket.write_all(resp.as_bytes()).await;
                });
            }
        });
        (format!("http://{addr}/v1"), handle)
    }

    fn find_header_end(req: &[u8]) -> Option<usize> {
        req.windows(4)
            .position(|w| w == b"\r\n\r\n")
            .map(|p| p + 4)
    }

    fn parse_content_length(head: &[u8]) -> usize {
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

    fn stub_client(base_url: &str) -> ApiOcrClient {
        let mut c = OcrConfig::default();
        c.api_base_url = base_url.to_string();
        c.api_model = "gpt-4o-mini".to_string();
        ApiOcrClient::from_config(&c).unwrap()
    }

    #[tokio::test]
    async fn api_ocr_transcribes_stub_image() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shot.webp");
        let img = image::DynamicImage::new_rgb8(4, 2);
        crate::hasher::save_webp(&img, &path, 80).unwrap();

        let (base_url, handle) = spawn_stub().await;
        let out = stub_client(&base_url).run_api_ocr(&path).await.unwrap();
        handle.abort();

        assert_eq!(out.full_text, "hello world");
        assert_eq!(out.word_count, 2);
        assert!(out.bounding_boxes.is_empty());
    }

    #[tokio::test]
    async fn api_health_and_models_hit_stub() {
        let (base_url, handle) = spawn_stub().await;
        let client = stub_client(&base_url);
        assert!(client.health_check().await.unwrap());
        assert_eq!(
            client.list_models().await.unwrap(),
            vec!["gpt-4o".to_string(), "gpt-4o-mini".to_string()]
        );
        handle.abort();
    }

    #[test]
    fn engine_from_config_maps_gateway() {
        assert_eq!(OcrEngine::from_config("gateway"), OcrEngine::Gateway);
        assert_eq!(OcrEngine::from_config("GATEWAY"), OcrEngine::Gateway);
    }

    #[test]
    fn gateway_client_needs_only_base_url() {
        let mut c = OcrConfig::default();
        assert!(GatewayOcrClient::from_config(&c).is_err());
        c.api_base_url = "https://gateway.local/v1/".to_string();
        let client = GatewayOcrClient::from_config(&c).unwrap();
        assert_eq!(client.root, "https://gateway.local");
    }

    #[test]
    fn quad_to_aabb_converts_and_rejects_degenerate() {
        let quad = serde_json::json!([[15.0, 5.0], [273.0, 5.0], [273.0, 28.0], [15.0, 28.0]]);
        assert_eq!(quad_to_aabb(&quad), Some((15, 5, 258, 23)));
        let flat = serde_json::json!([[0.0, 0.0], [10.0, 0.0], [10.0, 0.0], [0.0, 0.0]]);
        assert_eq!(quad_to_aabb(&flat), None);
        let bad = serde_json::json!([[0.0, 0.0], [1.0, 1.0]]);
        assert_eq!(quad_to_aabb(&bad), None);
    }

    #[test]
    fn multipart_body_carries_file_field_and_bytes() {
        let (content_type, body) = encode_multipart_file(b"PAYLOAD");
        assert!(content_type.starts_with("multipart/form-data; boundary="));
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("name=\"file\""));
        assert!(text.contains("filename=\"screenshot.webp\""));
        assert!(text.contains("PAYLOAD"));
    }

    #[test]
    fn parse_gateway_response_zips_lines_to_boxes() {
        let v = serde_json::json!({
            "text": "first line\nsecond line",
            "confidence": 0.9,
            "boxes": [
                [[0.0, 0.0], [10.0, 0.0], [10.0, 8.0], [0.0, 8.0]],
                [[0.0, 10.0], [20.0, 10.0], [20.0, 18.0], [0.0, 18.0]]
            ]
        });
        let out = parse_gateway_response(&v).unwrap();
        assert_eq!(out.full_text, "first line\nsecond line");
        assert_eq!(out.word_count, 4);
        assert_eq!(out.bounding_boxes.len(), 2);
        assert_eq!(out.bounding_boxes[0].text_content, "first line");
        assert_eq!(out.bounding_boxes[1].text_content, "second line");
        assert_eq!((out.bounding_boxes[1].x, out.bounding_boxes[1].width), (0, 20));
    }

    /// Gateway stub: `GET /health` → ok, `POST /ocr` → two quads over two
    /// text lines. Multipart bodies are binary-opaque here — the stub only
    /// needs the request line, but still drains `Content-Length` bytes.
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
                            if let Some(p) = find_header_end(&req) {
                                header_len = Some(p);
                                content_len = parse_content_length(&req[..p]);
                            }
                        }
                        if req.len() > 8_000_000 {
                            break;
                        }
                    }
                    let head = String::from_utf8_lossy(&req);
                    let first_line = head.lines().next().unwrap_or("");
                    let body = if first_line.starts_with("POST /ocr") {
                        "{\"text\":\"hello\\nworld\",\"confidence\":0.9,\"lines\":2,\"boxes\":[[[0,0],[10,0],[10,8],[0,8]],[[0,10],[20,10],[20,18],[0,18]]]}"
                    } else {
                        "{\"status\":\"ok\"}"
                    };
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = socket.write_all(resp.as_bytes()).await;
                });
            }
        });
        (format!("http://{addr}"), handle)
    }

    #[tokio::test]
    async fn gateway_ocr_transcribes_with_boxes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shot.webp");
        let img = image::DynamicImage::new_rgb8(4, 2);
        crate::hasher::save_webp(&img, &path, 80).unwrap();

        let (root, handle) = spawn_gateway_stub().await;
        let mut c = OcrConfig::default();
        c.api_base_url = root;
        let client = GatewayOcrClient::from_config(&c).unwrap();
        assert!(client.health_check().await.unwrap());
        let out = client.run_gateway_ocr(&path).await.unwrap();
        handle.abort();

        assert_eq!(out.full_text, "hello\nworld");
        assert_eq!(out.word_count, 2);
        assert_eq!(out.bounding_boxes.len(), 2);
        assert_eq!(out.bounding_boxes[0].text_content, "hello");
    }
}
