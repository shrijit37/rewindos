import { describe, it, expect } from "vitest";
import { setupPatch, setupProfile, SETUP_PROFILES } from "./setup-profiles";
import type { AppConfig } from "@/lib/config";

function cfg(over: {
  ocr?: Partial<AppConfig["ocr"]>;
  semantic?: Partial<AppConfig["semantic"]>;
  chat?: Partial<AppConfig["chat"]>;
} = {}): AppConfig {
  return {
    ocr: {
      enabled: true,
      engine: "tesseract",
      tesseract_lang: "eng",
      max_workers: 2,
      model_dir: "",
      python_bin: "python3",
      idle_timeout_secs: 60,
      api_base_url: "",
      api_key: "",
      api_model: "",
      api_timeout_secs: 120,
      api_max_width: 1600,
      ...over.ocr,
    },
    semantic: {
      enabled: true,
      ollama_url: "http://localhost:11434",
      model: "nomic-embed-text",
      embedding_dimensions: 768,
      provider: "ollama",
      api_base_url: "",
      api_key: "",
      api_model: "",
      ...over.semantic,
    },
    chat: {
      enabled: true,
      provider: "ollama",
      base_url: "http://localhost:11434/v1",
      api_key: "",
      model: "qwen2.5:7b",
      max_context_tokens: 8192,
      max_history_messages: 20,
      temperature: 0.7,
      agentic_tools: false,
      ...over.chat,
    },
  } as AppConfig;
}

const NO_WHISPER = { whisperPresent: false };

describe("setupPatch — lean", () => {
  it("turns off ocr, semantic, chat and meetings on an unconfigured config", () => {
    const p = setupPatch("lean", cfg(), NO_WHISPER);
    expect(p.setup.profile).toBe("lean");
    expect(p.ocr?.enabled).toBe(false);
    expect(p.semantic?.enabled).toBe(false);
    expect(p.chat?.enabled).toBe(false);
    expect(p.meeting?.enabled).toBe(false);
    // The usage ping is not a Lean/Balanced concern.
    expect(p.privacy).toBeUndefined();
  });

  it("keeps a configured remote OCR endpoint on and never writes engine", () => {
    const p = setupPatch(
      "lean",
      cfg({ ocr: { engine: "api", api_base_url: "https://api.openai.com/v1" } }),
      NO_WHISPER,
    );
    expect(p.ocr?.enabled).toBe(true);
    expect(p.ocr).not.toHaveProperty("engine");
  });

  it("turns OCR off when the engine is remote but no endpoint is configured", () => {
    const p = setupPatch("lean", cfg({ ocr: { engine: "api", api_base_url: "   " } }), NO_WHISPER);
    expect(p.ocr?.enabled).toBe(false);
  });

  it("switches semantic to api only when a remote embedding endpoint is complete", () => {
    const complete = setupPatch(
      "lean",
      cfg({ semantic: { api_base_url: "https://api.openai.com/v1", api_model: "text-embedding-3-small" } }),
      NO_WHISPER,
    );
    expect(complete.semantic).toEqual({ enabled: true, provider: "api" });

    const noModel = setupPatch(
      "lean",
      cfg({ semantic: { api_base_url: "https://api.openai.com/v1" } }),
      NO_WHISPER,
    );
    expect(noModel.semantic).toEqual({ enabled: false });
  });

  it("keeps chat on for a cloud provider with a key, off for a local one", () => {
    const cloud = setupPatch(
      "lean",
      cfg({ chat: { provider: "openai", api_key: "sk-test" } }),
      NO_WHISPER,
    );
    expect(cloud.chat?.enabled).toBe(true);

    const local = setupPatch("lean", cfg({ chat: { provider: "ollama", api_key: "sk-test" } }), NO_WHISPER);
    expect(local.chat?.enabled).toBe(false);

    const keyless = setupPatch("lean", cfg({ chat: { provider: "openrouter", api_key: "" } }), NO_WHISPER);
    expect(keyless.chat?.enabled).toBe(false);
  });
});

describe("setupPatch — balanced", () => {
  it("forces local tesseract + ollama and disables chat and meetings", () => {
    const p = setupPatch(
      "balanced",
      cfg({ ocr: { engine: "api", api_base_url: "https://api.openai.com/v1" } }),
      { whisperPresent: true },
    );
    expect(p.setup.profile).toBe("balanced");
    expect(p.ocr).toEqual({ enabled: true, engine: "tesseract" });
    expect(p.semantic).toEqual({ enabled: true, provider: "ollama" });
    expect(p.chat?.enabled).toBe(false);
    expect(p.meeting?.enabled).toBe(false);
    // Remote endpoint fields are left intact so switching back is one toggle.
    expect(p.ocr).not.toHaveProperty("api_base_url");
    expect(p.semantic).not.toHaveProperty("api_base_url");
    expect(p.chat).not.toHaveProperty("provider");
  });
});

describe("setupPatch — full", () => {
  it("forces local providers, disables the heartbeat, and follows whisper presence", () => {
    const withModel = setupPatch("full", cfg(), { whisperPresent: true });
    expect(withModel.setup.profile).toBe("full");
    expect(withModel.ocr).toEqual({ enabled: true, engine: "tesseract" });
    expect(withModel.semantic).toEqual({ enabled: true, provider: "ollama" });
    expect(withModel.chat).toEqual({
      enabled: true,
      provider: "ollama",
      base_url: "http://localhost:11434/v1",
    });
    expect(withModel.privacy).toEqual({ usage_heartbeat: false });
    expect(withModel.meeting?.enabled).toBe(true);

    // No model on disk → meetings stay off rather than failing at record time.
    expect(setupPatch("full", cfg(), NO_WHISPER).meeting?.enabled).toBe(false);
  });

  it("never clears an existing chat api key", () => {
    const p = setupPatch("full", cfg({ chat: { provider: "openai", api_key: "sk-keep" } }), NO_WHISPER);
    expect(p.chat).not.toHaveProperty("api_key");
  });
});

describe("setupProfile", () => {
  it("resolves every id in SETUP_PROFILES to its descriptor", () => {
    for (const p of SETUP_PROFILES) {
      expect(setupProfile(p.id)).toBe(p);
      expect(p.highlights).toHaveLength(3);
      expect(p.tagline.length).toBeGreaterThan(0);
    }
  });

  it("throws on an id outside the set", () => {
    expect(() => setupProfile("turbo" as never)).toThrow();
  });
});
