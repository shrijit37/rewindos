import { useState } from "react";
import type { CaptureVerdict, VerdictCode } from "../deriveCaptureVerdict";
import { setupProfile, type SetupProfileId } from "../setup-profiles";
import { Button } from "@/components/ui/button";

const CLOSING: Record<VerdictCode, string> = {
  working: "You're all set — capture is working.",
  unfiltered:
    "You're set — but note capture is running unfiltered; app and incognito exclusions aren't being enforced.",
  "gnome-degraded":
    "You're set, but app tracking is off until you install the 'Window Calls Extended' extension.",
  "paused-privacy":
    "You're set, but capture is paused until app tracking can be enforced — install the extension or capture anyway.",
  unsupported:
    "RewindOS can't capture on an X11 session — log out and pick a Wayland session at the login screen, then come back.",
  offline:
    "Almost there — the capture daemon isn't running yet. The status banner will keep nagging until it is.",
  stalled:
    "Almost there — capture isn't producing frames yet. The status banner will keep nagging until it is.",
  "paused-user": "You're set — capture is paused; resume it whenever you're ready.",
  "paused-locked": "You're set — capture will run whenever your session is unlocked.",
  waiting: "You're set — RewindOS is waiting for its first frame.",
  checking: "You're set — finishing the first capture check.",
  unknown: "You're set.",
};

interface FinishStepProps {
  verdict: CaptureVerdict;
  onStartTour: () => void;
  profile: SetupProfileId | null;
  restartNeeded: boolean;
  onRestartDaemon: () => Promise<void>;
}

export function FinishStep({
  verdict,
  onStartTour,
  profile,
  restartNeeded,
  onRestartDaemon,
}: FinishStepProps) {
  const [restart, setRestart] = useState<"idle" | "pending" | "done" | "error">("idle");
  const [restartError, setRestartError] = useState<string | null>(null);
  const chosen = profile ? setupProfile(profile) : null;

  const restartDaemonNow = async () => {
    setRestart("pending");
    setRestartError(null);
    try {
      await onRestartDaemon();
      setRestart("done");
    } catch (e) {
      setRestart("error");
      setRestartError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-xl font-semibold text-text-primary">You're ready</h2>
      {chosen && (
        <p className="text-sm text-text-secondary">
          Setup: <span className="font-medium text-text-primary">{chosen.name}</span> —{" "}
          {chosen.tagline}
        </p>
      )}

      <p className="text-sm leading-relaxed text-text-secondary">{CLOSING[verdict.code]}</p>

      <div className="flex flex-col gap-2 rounded-md border border-border/50 bg-background/40 p-4 text-sm text-text-secondary">
        <p className="font-medium text-text-primary">See what RewindOS can do</p>
        <p className="text-xs text-text-muted">
          Six quick stops — search, ask, history, journal, vault export, and AI setup.
        </p>
        <div>
          <Button size="sm" onClick={onStartTour}>
            Take the tour (1 min)
          </Button>
        </div>
      </div>


      <div className="flex flex-col gap-2 rounded-md border border-border/50 bg-background/40 p-4 text-sm text-text-secondary">
        <p>
          Press{" "}
          <kbd className="rounded border border-border/60 bg-surface px-1.5 py-0.5 font-mono text-[11px] text-text-primary">
            Ctrl+Shift+Space
          </kbd>{" "}
          anytime to search everything you've seen.
        </p>
        <p className="text-xs text-text-muted">
          RewindOS lives in your system tray. Open Settings from there to tune capture,
          privacy, and AI features.
        </p>
      </div>
      {restartNeeded && chosen && (
        <div className="flex flex-col gap-2 rounded-md border border-border/50 bg-background/40 p-4 text-sm text-text-secondary">
          <p className="leading-relaxed">
            RewindOS applies the capture pipeline immediately. OCR, semantic search, chat and
            meetings are read when the daemon starts — restart it to pick up {chosen.name}.
          </p>
          <div>
            <Button size="sm" onClick={() => void restartDaemonNow()} disabled={restart === "pending"}>
              Restart daemon
            </Button>
          </div>
          {restart !== "idle" && (
            <p className="font-mono text-[11px] text-text-muted">
              {restart === "pending" && "Restarting…"}
              {restart === "done" && "Daemon restarted"}
              {restart === "error" && (restartError ?? "restart failed")}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
