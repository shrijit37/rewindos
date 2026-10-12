import { useQuery } from "@tanstack/react-query";
import { ocrHealthCheck, semanticHealthCheck, whisperModelPresent } from "@/lib/api";
import { queryKeys } from "@/lib/query-keys";
import { useConfig } from "@/features/settings/hooks/useConfig";
import { SETUP_PROFILES, type SetupProfileId } from "../setup-profiles";

interface SetupStepProps {
  selected: SetupProfileId | null;
  onSelect: (id: SetupProfileId) => void;
}

export function SetupStep({ selected, onSelect }: SetupStepProps) {
  const { config } = useConfig();

  // Both probes force the LOCAL engine/provider in the payload: a user with
  // ocr.engine === "api" must not have their screenshots uploaded just by
  // opening the wizard.
  const ocr = useQuery({
    queryKey: queryKeys.ocrHealth("tesseract"),
    queryFn: () => {
      if (!config) return false;
      return ocrHealthCheck({ ...config.ocr, engine: "tesseract" });
    },
    enabled: config !== null,
    staleTime: 30_000,
    retry: false,
  });

  const ollama = useQuery({
    queryKey: queryKeys.semanticHealth("ollama"),
    enabled: config !== null,
    queryFn: () => {
      if (!config) return false;
      return semanticHealthCheck({ ...config.semantic, provider: "ollama" });
    },
    staleTime: 30_000,
    retry: false,
  });

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h2 className="text-xl font-semibold text-text-primary">
          How much should run on your machine?
        </h2>
        <p className="text-sm leading-relaxed text-text-secondary">
          Pick a setup — everything stays changeable later in Settings.
        </p>
      </div>

      <div className="flex max-h-[22rem] flex-col gap-3 overflow-y-auto pr-1">
        {SETUP_PROFILES.map((p) => {
          const isSelected = selected === p.id;
          return (
            <button
              key={p.id}
              type="button"
              aria-pressed={isSelected}
              onClick={() => onSelect(p.id)}
              className={`flex flex-col gap-2 rounded-md border p-4 text-left transition-colors ${
                isSelected
                  ? "border-accent bg-accent/5"
                  : "border-border/50 bg-background/40 hover:border-border"
              }`}
            >
              <div className="flex flex-col gap-0.5">
                <span className="text-sm font-medium text-text-primary">{p.name}</span>
                <span className="text-xs text-text-secondary">{p.tagline}</span>
              </div>

              <ul className="flex list-disc flex-col gap-1 pl-4 text-xs leading-relaxed text-text-secondary">
                {p.highlights.map((h) => (
                  <li key={h}>{h}</li>
                ))}
              </ul>

              <div className="flex flex-col gap-0.5 border-t border-border/40 pt-2">
                {p.id === "balanced" && (
                  <ReadinessLine
                    query={ocr}
                    ok="Tesseract found"
                    missing="Tesseract not found — OCR will stay quiet"
                  />
                )}
                {(p.id === "balanced" || p.id === "full") && (
                  <ReadinessLine
                    query={ollama}
                    ok="Ollama found"
                    missing="Ollama not found — start it before the daemon picks this up"
                  />
                )}
                {p.id === "full" && <WhisperLine />}
                <p className="text-[11px] leading-relaxed text-text-muted">{p.note}</p>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function ReadinessLine({
  query,
  ok,
  missing,
}: {
  query: { data: boolean | undefined; isPending: boolean };
  ok: string;
  missing: string;
}) {
  const text = query.isPending ? "checking…" : query.data === true ? ok : missing;
  const tone =
    query.data === true
      ? "text-signal-active"
      : query.isPending
        ? "text-text-muted"
        : "text-signal-paused";
  return <p className={`font-mono text-[11px] ${tone}`}>{text}</p>;
}

function WhisperLine() {
  const { data, isPending } = useQuery({
    queryKey: queryKeys.whisperModel(),
    queryFn: whisperModelPresent,
    staleTime: 10_000,
    retry: false,
  });
  const text =
    isPending
      ? "checking…"
      : data === true
        ? "Whisper model ready"
        : "Whisper model missing — meeting recording stays off until you download it in Settings → Meetings";
  return (
    <p className={`font-mono text-[11px] ${data === true ? "text-signal-active" : "text-signal-paused"}`}>
      {text}
    </p>
  );
}