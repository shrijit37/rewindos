import type { AppConfig } from "@/lib/config";

export type SetupProfileId = "lean" | "balanced" | "full";

export interface SetupProfile {
  id: SetupProfileId;
  /** Display name: "Lean" | "Balanced" | "Full". */
  name: string;
  /** One line, shown under the name. */
  tagline: string;
  /** Three bullets, shown as a list. */
  highlights: string[];
  /** Footer line: requirements / privacy cost / limits. */
  note: string;
}

/** Single source of truth for the wizard copy and the config mapping. */
export const SETUP_PROFILES: SetupProfile[] = [
  {
    id: "lean",
    name: "Lean",
    tagline: "Capture here, intelligence off-machine.",
    highlights: [
      "No OCR, embeddings, chat or transcription run on this machine — the lightest option for a low-end PC.",
      "Remote endpoints you've already connected keep working; anything unconnected stays off.",
      "Nothing is downloaded and no local model is loaded.",
    ],
    note: "When a remote endpoint is in use, screenshots and text leave this machine — that's the trade for the idle CPU.",
  },
  {
    id: "balanced",
    name: "Balanced",
    tagline: "Local OCR and semantic search.",
    highlights: [
      "Text search and meaning-aware recall run here through Ollama.",
      "No chat model, no meeting recording — modest CPU and memory.",
      "Nothing is sent anywhere.",
    ],
    note: "Needs Tesseract installed and Ollama running. RewindOS pulls the embedding model on the first daemon start.",
  },
  {
    id: "full",
    name: "Full",
    tagline: "Everything local, works offline.",
    highlights: [
      "Balanced, plus on-device chat and meeting transcripts.",
      "The anonymous usage ping is switched off.",
      "Saved cloud endpoints stay saved but unused — switch back any time in Settings.",
    ],
    note: "Screenshots, text and prompts never leave this machine. Update checks still reach the release feed.",
  },
];

export function setupProfile(id: SetupProfileId): SetupProfile {
  const found = SETUP_PROFILES.find((p) => p.id === id);
  // SETUP_PROFILES covers every id in the union, so a miss can only mean the
  // two drifted apart — fail loudly rather than render an empty card.
  if (!found) throw new Error(`unknown setup profile: ${id}`);
  return found;
}

export type SetupPatch = {
  setup: { profile: SetupProfileId };
  ocr?: Partial<AppConfig["ocr"]>;
  semantic?: Partial<AppConfig["semantic"]>;
  chat?: Partial<AppConfig["chat"]>;
  meeting?: Partial<AppConfig["meeting"]>;
  privacy?: Partial<AppConfig["privacy"]>;
};

const REMOTE_CHAT_PROVIDERS = ["openai", "openrouter", "custom"];

/**
 * Map a setup type onto the config sections it controls. Every remote decision
 * is derived from `current` — a patch never invents a credential or a provider
 * the user did not already configure.
 */
export function setupPatch(
  id: SetupProfileId,
  current: AppConfig,
  opts: { whisperPresent: boolean },
): SetupPatch {
  if (id === "lean") {
    // A remote OCR path stays alive only when one is genuinely configured.
    const remoteOcr = (current.ocr.engine === "api" || current.ocr.engine === "gateway")
      && current.ocr.api_base_url.trim() !== "";
    const remoteEmbeddings = current.semantic.api_base_url.trim() !== ""
      && current.semantic.api_model.trim() !== "";
    const remoteChat = current.chat.api_key.trim() !== ""
      && REMOTE_CHAT_PROVIDERS.includes(current.chat.provider);

    return {
      setup: { profile: "lean" },
      ocr: remoteOcr ? { enabled: true } : { enabled: false },
      semantic: remoteEmbeddings ? { enabled: true, provider: "api" } : { enabled: false },
      chat: remoteChat ? { enabled: true } : { enabled: false },
      // No remote transcription path exists, so recording is always off.
      meeting: { enabled: false },
    };
  }

  if (id === "balanced") {
    // api_* fields are deliberately untouched: switching back is one toggle.
    return {
      setup: { profile: "balanced" },
      ocr: { enabled: true, engine: "tesseract" },
      semantic: { enabled: true, provider: "ollama" },
      chat: { enabled: false },
      meeting: { enabled: false },
    };
  }

  return {
    setup: { profile: "full" },
    ocr: { enabled: true, engine: "tesseract" },
    semantic: { enabled: true, provider: "ollama" },
    // api_key is left in place: it is inert against localhost, and clearing it
    // would destroy a credential the user may want back.
    chat: { enabled: true, provider: "ollama", base_url: "http://localhost:11434/v1" },
    meeting: { enabled: opts.whisperPresent },
    privacy: { usage_heartbeat: false },
  };
}
