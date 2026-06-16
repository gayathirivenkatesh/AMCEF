from typing import Optional, Dict, Any


class AMCEF:
    """
    Work-3 Grade AMCEF Fusion Engine

    Features
    --------
    ✔ Adaptive confidence-based gating
    ✔ Modality dominance logic
    ✔ Neutral suppression
    ✔ Emotion similarity soft fusion
    ✔ Explicit fusion weights (Caudio, Cvideo)
    ✔ Emotion Reliability Index (ERI)
    ✔ Explainable rule trace
    ✔ Robust to missing modalities
    """

    CONFIDENCE_THRESHOLD = 0.25
    CLOSE_CONFIDENCE_DIFF = 0.12

    # psychologically close emotion groups
    EMOTION_GROUPS = [
        {"happy", "neutral"},
        {"sad", "neutral"},
        {"fear", "surprise"},
        {"angry", "disgust"}
    ]

    # --------------------------------------------------
    # Similarity
    # --------------------------------------------------
    @staticmethod
    def _same_group(e1: Optional[str], e2: Optional[str]) -> bool:
        if not e1 or not e2:
            return False
        return any(e1 in g and e2 in g for g in AMCEF.EMOTION_GROUPS)

    # --------------------------------------------------
    # Adaptive Weights (Work-3 core math)
    # --------------------------------------------------
    @staticmethod
    def _adaptive_weights(a: float, v: float) -> tuple[float, float]:
        total = a + v + 1e-6
        return round(a / total, 3), round(v / total, 3)

    # --------------------------------------------------
    # ERI
    # --------------------------------------------------
    @staticmethod
    def _compute_eri(a: float, v: float, source: str, agreement: bool):
        base = max(a, v) * 0.6

        if source in {"fusion", "soft_fusion"}:
            base = (a + v) / 2

        if agreement:
            base += 0.1

        if source in {"low_confidence_fallback", "none"}:
            base -= 0.15

        eri = round(max(0, min(1, base)), 2)

        level = (
            "HIGH" if eri >= 0.7
            else "MODERATE" if eri >= 0.4
            else "LOW"
        )

        return eri, level

    # --------------------------------------------------
    # Fusion
    # --------------------------------------------------
    @staticmethod
    def fuse(
        audio_emotion: Optional[str],
        audio_conf: float,
        video_emotion: Optional[str],
        video_conf: float
    ) -> Dict[str, Any]:

        trace = []
        agreement = (
            audio_emotion == video_emotion or
            AMCEF._same_group(audio_emotion, video_emotion)
        )

        # ---------- Missing modality ----------
        if audio_emotion is None and video_emotion is None:
            eri, lvl = AMCEF._compute_eri(0, 0, "none", False)
            return {
                "fused_emotion": "neutral",
                "fused_confidence": 0.0,
                "source": "none",
                "weights": {"audio": 0, "video": 0},
                "rule_trace": ["no_modality_available"],
                "explanation": "No modality available.",
                "eri_score": eri,
                "eri_level": lvl
            }

        if audio_emotion is None:
            eri, lvl = AMCEF._compute_eri(0, video_conf, "video", False)
            return {
                "fused_emotion": video_emotion,
                "fused_confidence": round(video_conf, 2),
                "source": "video",
                "weights": {"audio": 0, "video": 1},
                "rule_trace": ["audio_missing"],
                "explanation": "Audio missing → video used.",
                "eri_score": eri,
                "eri_level": lvl
            }

        if video_emotion is None:
            eri, lvl = AMCEF._compute_eri(audio_conf, 0, "audio", False)
            return {
                "fused_emotion": audio_emotion,
                "fused_confidence": round(audio_conf, 2),
                "source": "audio",
                "weights": {"audio": 1, "video": 0},
                "rule_trace": ["video_missing"],
                "explanation": "Video missing → audio used.",
                "eri_score": eri,
                "eri_level": lvl
            }

        # ---------- Adaptive weights ----------
        wa, wv = AMCEF._adaptive_weights(audio_conf, video_conf)
        trace.append(f"weights_audio={wa}")
        trace.append(f"weights_video={wv}")

        # ---------- Both weak ----------
        if audio_conf < AMCEF.CONFIDENCE_THRESHOLD and video_conf < AMCEF.CONFIDENCE_THRESHOLD:
            chosen = audio_emotion if audio_conf >= video_conf else video_emotion
            eri, lvl = AMCEF._compute_eri(audio_conf, video_conf, "low_confidence_fallback", False)

            return {
                "fused_emotion": chosen,
                "fused_confidence": round(max(audio_conf, video_conf), 2),
                "source": "low_confidence_fallback",
                "weights": {"audio": wa, "video": wv},
                "rule_trace": trace + ["both_low_conf"],
                "explanation": "Both modalities weak → fallback to stronger.",
                "eri_score": eri,
                "eri_level": lvl
            }

        # ---------- Agreement ----------
        if agreement:
            fused_conf = min(1.0, (audio_conf * wa + video_conf * wv) + 0.1)
            eri, lvl = AMCEF._compute_eri(audio_conf, video_conf, "fusion", True)

            return {
                "fused_emotion": audio_emotion,
                "fused_confidence": round(fused_conf, 2),
                "source": "fusion",
                "weights": {"audio": wa, "video": wv},
                "rule_trace": trace + ["agreement_boost"],
                "explanation": "Modalities agree → weighted fusion + boost.",
                "eri_score": eri,
                "eri_level": lvl
            }

        # ---------- Neutral suppression ----------
        if audio_emotion == "neutral" and video_conf > audio_conf + 0.05:
            eri, lvl = AMCEF._compute_eri(audio_conf, video_conf, "video_override", False)
            return {
                "fused_emotion": video_emotion,
                "fused_confidence": round(video_conf, 2),
                "source": "video_override",
                "weights": {"audio": wa, "video": wv},
                "rule_trace": trace + ["neutral_suppressed_audio"],
                "explanation": "Neutral audio suppressed.",
                "eri_score": eri,
                "eri_level": lvl
            }

        if video_emotion == "neutral" and audio_conf > video_conf + 0.05:
            eri, lvl = AMCEF._compute_eri(audio_conf, video_conf, "audio_override", False)
            return {
                "fused_emotion": audio_emotion,
                "fused_confidence": round(audio_conf, 2),
                "source": "audio_override",
                "weights": {"audio": wa, "video": wv},
                "rule_trace": trace + ["neutral_suppressed_video"],
                "explanation": "Neutral video suppressed.",
                "eri_score": eri,
                "eri_level": lvl
            }

        # ---------- Confidence dominance ----------
        diff = abs(audio_conf - video_conf)

        if diff > AMCEF.CLOSE_CONFIDENCE_DIFF:
            src = "audio" if audio_conf > video_conf else "video"
            chosen = audio_emotion if src == "audio" else video_emotion
            conf = max(audio_conf, video_conf)

            eri, lvl = AMCEF._compute_eri(audio_conf, video_conf, src, False)

            return {
                "fused_emotion": chosen,
                "fused_confidence": round(conf, 2),
                "source": src,
                "weights": {"audio": wa, "video": wv},
                "rule_trace": trace + ["confidence_dominance"],
                "explanation": f"{src} confidence higher.",
                "eri_score": eri,
                "eri_level": lvl
            }

        # ---------- Soft fusion ----------
        fused_conf = (audio_conf * wa + video_conf * wv)
        eri, lvl = AMCEF._compute_eri(audio_conf, video_conf, "soft_fusion", False)

        return {
            "fused_emotion": audio_emotion if audio_conf >= video_conf else video_emotion,
            "fused_confidence": round(fused_conf, 2),
            "source": "soft_fusion",
            "weights": {"audio": wa, "video": wv},
            "rule_trace": trace + ["soft_fusion"],
            "explanation": "Close confidence → soft weighted fusion.",
            "eri_score": eri,
            "eri_level": lvl
        }
    # --------------------------------------------------
    # Weighted Multimodal Fusion (Step 5)
    # --------------------------------------------------
    @staticmethod
    def fuse_weighted(
        audio_emotion: str,
        audio_conf: float,
        video_emotion: str,
        video_conf: float
    ) -> Dict[str, Any]:
        """
        FinalScore = 0.6 * AudioConfidence + 0.4 * VideoConfidence
        """
        wa = 0.6
        wv = 0.4
        
        final_score = wa * audio_conf + wv * video_conf
        
        print(f"DEBUG FUSE: ae={audio_emotion}, ac={audio_conf}, ve={video_emotion}, vc={video_conf}")
        
        # Decision logic:
        # 1. If both agree, that's our winner (and we boost confidence)
        if audio_emotion == video_emotion:
            fused_emo = audio_emotion
            final_score = min(1.0, final_score * 1.2)
        
        # 2. If one is fallback, pick the other if it's not neutral
        elif audio_conf <= 0.2 and video_emotion != "neutral":
            fused_emo = video_emotion
        elif video_conf <= 0.25 and audio_emotion != "neutral":
            fused_emo = audio_emotion
            
        # 3. Weighted choice
        elif (wa * audio_conf) >= (wv * video_conf):
            fused_emo = audio_emotion
        else:
            fused_emo = video_emotion

        # 4. Global Neutral override: If fused is neutral but one side is very strong on something else...
        if fused_emo == "neutral":
            if video_emotion != "neutral" and video_conf > 0.4:
                fused_emo = video_emotion
            elif audio_emotion != "neutral" and audio_conf > 0.4:
                fused_emo = audio_emotion

        print(f"DEBUG FUSE: chosen_emo={fused_emo}, final_score={final_score}")

        return {
            "fused_emotion": fused_emo,
            "fused_confidence": round(final_score, 3),
            "source": "weighted_fusion",
            "weights": {"audio": wa, "video": wv},
            "rule_trace": ["weighted_multimodal_sum"],
            "explanation": f"Multimodal Fusion: Audio({wa}) + Video({wv})",
            "eri_score": round(final_score, 2),
            "eri_level": "HIGH" if final_score >= 0.7 else "MODERATE" if final_score >= 0.4 else "LOW"
        }
