"""Export a train_paligemma state-dict checkpoint to a save_pretrained dir
loadable by SWMGradModel (their planning stack expects HF-style dirs)."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from train_paligemma import PaliGemmaWMTrainable

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--processor-path", required=True)
ap.add_argument("--base-model", default="google/paligemma-3b-pt-224")
ap.add_argument("--out", required=True)
args = ap.parse_args()

m = PaliGemmaWMTrainable(args.processor_path, args.base_model, device="cpu")
sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
m.model.load_state_dict(sd["model"])
m.model.to(torch.bfloat16).save_pretrained(args.out)
m.processor.save_pretrained(args.out)
torch.save({"action_mean": sd["action_mean"], "action_std": sd["action_std"]},
           Path(args.out) / "action_stats.pt")
print(f"exported epoch-{sd.get('epoch')} -> {args.out}")
