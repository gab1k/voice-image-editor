from pathlib import Path
from typing import Optional, Union
import sys
import torch


class ASRModelWrapper:
    """
    Wrapper for ASR models:
      - GigaAM-v3
      - T-one
      - NVIDIA Parakeet-TDT-0.6B-v3
    """

    SUPPORTED_TYPES = ("gigaam_v3", "tone", "parakeet")

    def __init__(
        self,
        model_type: str,
        device: Optional[str] = None,
    ) -> None:
        if model_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unknown model_type='{model_type}'")

        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Initialization of ASRModelWrapper: {model_type}")
        self._load_model()

    def _load_model(self) -> None:
        if self.model_type == "gigaam_v3":
            self._load_gigaam()
        elif self.model_type == "tone":
            self._load_tone()
        elif self.model_type == "parakeet":
            self._load_parakeet()

    def _load_gigaam(self) -> None:
        from transformers import AutoModel

        revision = "e2e_rnnt"
        self.model = AutoModel.from_pretrained(
            "ai-sage/GigaAM-v3",
            revision=revision,
            trust_remote_code=True,
        )
        print("GigaAM-v3 loaded")

    def _load_tone(self) -> None:
        tone_repo_path = Path("/tmp/T-one")
        if not tone_repo_path.exists():
            import subprocess
            subprocess.run(
                ["git", "clone", "https://github.com/voicekit-team/T-one.git", str(tone_repo_path)],
                check=True,
            )

        if str(tone_repo_path) not in sys.path:
            sys.path.insert(0, str(tone_repo_path))

        from tone import StreamingCTCPipeline

        self.pipeline = StreamingCTCPipeline.from_hugging_face()
        print("T-one loaded")

    def _load_parakeet(self) -> None:
        import nemo.collections.asr as nemo_asr
        import os

        original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        try:
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name="nvidia/parakeet-tdt-0.6b-v3"
            )
        finally:
            if original_cuda_visible is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        
        self.model.eval()

        if hasattr(self.model, "decoding"):
            if hasattr(self.model.decoding, "decoding_computer"):
                dc = self.model.decoding.decoding_computer
                if hasattr(dc, "cuda_graphs_mode"):
                    dc.cuda_graphs_mode = None
            if hasattr(self.model.decoding, "cfg"):
                if hasattr(self.model.decoding.cfg, "cuda_graphs_mode"):
                    self.model.decoding.cfg.cuda_graphs_mode = None

        if self.device == "cpu":
            self.model = self.model.cpu()
        elif self.device.startswith("cuda"):
            self.model = self.model.to(self.device)

        print("Parakeet-TDT-0.6B-v3 loaded")

    def transcribe(self, audio_input: Union[str, Path]) -> str:
        audio_input = str(audio_input)

        if self.model_type == "gigaam_v3":
            return self._transcribe_gigaam(audio_input)
        elif self.model_type == "tone":
            return self._transcribe_tone(audio_input)
        elif self.model_type == "parakeet":
            return self._transcribe_parakeet(audio_input)

        raise RuntimeError("Unsupported ASR model")

    def _transcribe_gigaam(self, audio_input: str) -> str:
        try:
            text = self.model.transcribe(audio_input)
        except ValueError as e:
            if "Too long" in str(e):
                text = self.model.transcribe_longform(audio_input)
            else:
                raise
                
        if isinstance(text, list):
            if text and isinstance(text[0], dict):
                text = " ".join(
                    item.get("text", item.get("transcription", str(item)))
                    for item in text
                )
            else:
                text = " ".join(str(item) for item in text)
        
        return text.lower()

    def _transcribe_tone(self, audio_input: str) -> str:
        from tone import read_audio

        audio = read_audio(audio_input)
        phrases = self.pipeline.forward_offline(audio)
        text = " ".join(p.text for p in phrases).strip()
        return text.lower()

    def _transcribe_parakeet(self, audio_input: str) -> str:
        outputs = self.model.transcribe(
            [audio_input],
            batch_size=1,
            num_workers=0,
        )
        text = getattr(outputs[0], "text", outputs[0])
        return text.lower()

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "pipeline"):
            del self.pipeline
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    asr = ASRModelWrapper(model_type="tone", device="cuda")
    print(asr.transcribe("sample.wav"))
    asr.close()
