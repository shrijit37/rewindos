import { useState } from "react";
import {
  SETUP_PROFILES,
  setupProfile,
  type SetupProfileId,
} from "@/features/onboarding/setup-profiles";
import { restartDaemon } from "@/lib/api";
import type { AppConfig } from "@/lib/config";
import { SectionTitle } from "../primitives/SectionTitle";
import { Field } from "../primitives/Field";
import { TextInput } from "../primitives/TextInput";
import { CategoryRulesEditor } from "../primitives/CategoryRulesEditor";
import { Button } from "@/components/ui/button";
import { useOnboarding } from "@/features/onboarding/OnboardingContext";
import { useTour } from "@/features/tour/TourContext";
import { UpdateSection } from "../UpdateSection";

interface TabProps {
  config: AppConfig;
  update: <S extends keyof AppConfig, K extends keyof AppConfig[S]>(
    section: S, key: K, value: AppConfig[S][K],
  ) => void;
}

export function GeneralTab({ config, update }: TabProps) {
  const [restart, setRestart] = useState<"idle" | "pending" | "done" | "error">("idle");
  const [restartError, setRestartError] = useState<string | null>(null);
  // config.setup.profile is a union, but a hand-edited config.toml could hold
  // anything — fall back rather than throw inside Settings.
  const active = SETUP_PROFILES.find((p) => p.id === config.setup?.profile);
  const profile: SetupProfileId = active ? active.id : "balanced";
  const descriptor = setupProfile(profile);

  const doRestart = async () => {
    setRestart("pending");
    setRestartError(null);
    try {
      await restartDaemon();
      setRestart("done");
    } catch (e) {
      setRestart("error");
      setRestartError(e instanceof Error ? e.message : String(e));
    }
  };
  const { open } = useOnboarding();
  const { start: startTour } = useTour();
  return (
    <>
      <SectionTitle>General</SectionTitle>
      <Field label="Global Hotkey">
        <TextInput
          value={config.ui.global_hotkey}
          onChange={(v) => update("ui", "global_hotkey", v)}
        />
      </Field>
      <Field label={`Setup · ${descriptor.name}`}>
        <div className="flex flex-col gap-2">
          <p className="text-xs text-text-muted">{descriptor.tagline}</p>
          <div>
            <Button variant="editorial" size="editorial" onClick={open}>
              Change setup
            </Button>
          </div>
        </div>
      </Field>
      <Field label="Daemon">
        <div className="flex flex-col gap-2">
          <div>
            <Button
              variant="editorial"
              size="editorial"
              onClick={() => void doRestart()}
              disabled={restart === "pending"}
            >
              Restart daemon
            </Button>
          </div>
          <p className="text-xs text-text-muted">
            OCR, semantic search, chat and meetings are read when the daemon starts.
          </p>
          {restart !== "idle" && (
            <span className="font-mono text-[11px] text-text-muted">
              {restart === "pending" && "Restarting…"}
              {restart === "done" && "Daemon restarted"}
              {restart === "error" && (restartError ?? "restart failed")}
            </span>
          )}
        </div>
      </Field>
      <Field label="Feature tour">
        <Button variant="editorial" size="editorial" onClick={startTour}>
          Replay feature tour
        </Button>
      </Field>
      <SectionTitle>Activity Categories</SectionTitle>
      <p className="text-xs text-text-muted -mt-2">
        Add app keywords to categories. These extend the built-in defaults.
      </p>
      <CategoryRulesEditor
        rules={config.categories.rules ?? {}}
        onChange={(rules) => update("categories", "rules", rules)}
      />

      <UpdateSection />
    </>
  );
}
