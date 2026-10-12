import { useState, useEffect } from "react";
import type { AppConfig } from "@/lib/config";
import { SectionTitle } from "../primitives/SectionTitle";
import { Field } from "../primitives/Field";
import { TextInput } from "../primitives/TextInput";
import { NumberInput } from "../primitives/NumberInput";
import { Toggle } from "../primitives/Toggle";
import { Button } from "@/components/ui/button";
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from "@/components/ui/select";
import { useQuery, useMutation } from "@tanstack/react-query";
import { ocrHealthCheck, ocrListModels } from "@/lib/api";
import { queryKeys } from "@/lib/query-keys";

function useDebounced<T>(value: T, ms: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return debounced;
}

interface TabProps {
  config: AppConfig;
  update: <S extends keyof AppConfig, K extends keyof AppConfig[S]>(
    section: S, key: K, value: AppConfig[S][K],
  ) => void;
}

export function OCRTab({ config, update }: TabProps) {
  const isApi = config.ocr.engine === "api";
  const [testResult, setTestResult] = useState<string | null>(null);

  const debouncedBaseUrl = useDebounced(config.ocr.api_base_url, 500);
  const debouncedApiKey = useDebounced(config.ocr.api_key, 500);

  const test = useMutation({
    mutationFn: () => ocrHealthCheck(config.ocr),
    onSuccess: (ok) => setTestResult(ok ? "connected" : "unreachable"),
    onError: (e) => setTestResult(String(e)),
  });

  const { data: models } = useQuery({
    queryKey: queryKeys.ocrModels(debouncedBaseUrl, debouncedApiKey),
    queryFn: () => ocrListModels({ ...config.ocr, api_base_url: debouncedBaseUrl, api_key: debouncedApiKey }),
    enabled: config.ocr.enabled && isApi && !!debouncedBaseUrl,
    retry: false,
    staleTime: 30_000,
  });

  return (
    <>
      <SectionTitle>OCR</SectionTitle>
      <Field label="Enabled">
        <Toggle
          checked={config.ocr.enabled}
          onChange={(v) => update("ocr", "enabled", v)}
        />
      </Field>
      <Field label="Engine">
        <Select value={config.ocr.engine} onValueChange={(v) => { update("ocr", "engine", v as AppConfig["ocr"]["engine"]); setTestResult(null); }}>
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="tesseract">Tesseract</SelectItem>
            <SelectItem value="paddleocr">PaddleOCR</SelectItem>
            <SelectItem value="api">API (cloud)</SelectItem>
          </SelectContent>
        </Select>
      </Field>
      {config.ocr.engine === "tesseract" && (
        <Field label="Tesseract Language">
          <TextInput
            value={config.ocr.tesseract_lang}
            onChange={(v) => update("ocr", "tesseract_lang", v)}
          />
        </Field>
      )}
      {config.ocr.engine === "paddleocr" && (
        <>
          <Field label="Python Binary">
            <TextInput
              value={config.ocr.python_bin}
              onChange={(v) => update("ocr", "python_bin", v)}
            />
          </Field>
          <Field label="Idle Timeout (seconds)">
            <NumberInput
              value={config.ocr.idle_timeout_secs}
              min={10}
              max={600}
              onChange={(v) => update("ocr", "idle_timeout_secs", v)}
            />
          </Field>
        </>
      )}
      {isApi && (
        <>
          <Field
            label="Base URL"
            hint="Any OpenAI-compatible vision endpoint (OpenAI, OpenRouter, self-hosted gateway). Takes effect after a daemon restart."
          >
            <TextInput
              value={config.ocr.api_base_url}
              onChange={(v) => { update("ocr", "api_base_url", v); setTestResult(null); }}
            />
          </Field>
          <Field label="API Key">
            <TextInput
              type="password"
              value={config.ocr.api_key}
              onChange={(v) => { update("ocr", "api_key", v); setTestResult(null); }}
              autoComplete="off"
            />
          </Field>
          <Field label="Model">
            <>
              <TextInput
                value={config.ocr.api_model}
                onChange={(v) => update("ocr", "api_model", v)}
                list="ocr-model-options"
              />
              <datalist id="ocr-model-options">
                {(models ?? []).map((m) => (
                  <option key={m} value={m} />
                ))}
              </datalist>
            </>
          </Field>
          <Field label="Timeout (seconds)">
            <NumberInput
              value={config.ocr.api_timeout_secs}
              min={5}
              max={300}
              onChange={(v) => update("ocr", "api_timeout_secs", v)}
            />
          </Field>
          <Field
            label="Max Upload Width (px)"
            hint="Screenshots wider than this are downscaled before upload."
          >
            <NumberInput
              value={config.ocr.api_max_width}
              min={640}
              max={3840}
              onChange={(v) => update("ocr", "api_max_width", v)}
            />
          </Field>
          <Field label="">
            <div className="flex items-center gap-3">
              <Button
                variant="editorial"
                size="editorial"
                onClick={() => test.mutate()}
                disabled={test.isPending}
              >
                {test.isPending ? "testing..." : "Test connection"}
              </Button>
              {testResult && (
                <span
                  className={`font-mono text-[11px] ${
                    testResult === "connected" ? "text-signal-active" : "text-text-muted"
                  }`}
                >
                  {testResult}
                </span>
              )}
            </div>
          </Field>
          <Field label="API mode note">
            <span className="font-mono text-[11px] text-text-muted">
              Transcription only — no word-level overlays; no local OCR binary is used.
            </span>
          </Field>
        </>
      )}
      <Field
        label="Max Workers"
        hint={isApi ? "Caps concurrent API requests (rate limits)." : undefined}
      >
        <NumberInput
          value={config.ocr.max_workers}
          min={1}
          max={8}
          onChange={(v) => update("ocr", "max_workers", v)}
        />
      </Field>
    </>
  );
}
