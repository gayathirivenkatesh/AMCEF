import sys
import os
import csv
from typing import List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fusion.amcef import AMCEF


def calculate_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true) if y_true else 0.0


def evaluate() -> None:
    print("--------------------------------------------------")
    print("AMCEF Evaluation Script with ERI Analysis")
    print("--------------------------------------------------")

    input_path = "assets/fusion_input.csv"
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    with open(input_path, 'r', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    # --------------------------------------------------
    # Containers
    # --------------------------------------------------
    y_true, y_audio, y_video, y_fusion = [], [], [], []

    fusion_helped = 0
    fusion_hurt = 0
    confusion = {}

    eri_scores = []
    eri_correct = {"HIGH": 0, "MODERATE": 0, "LOW": 0}
    eri_total = {"HIGH": 0, "MODERATE": 0, "LOW": 0}

    enriched_rows = []

    # --------------------------------------------------
    # Main Loop
    # --------------------------------------------------
    for row in rows:
        gt = row["gt_emotion"]
        a_pred = row["audio_emotion"]
        v_pred = row["video_emotion"]

        try:
            a_conf = float(row["audio_conf"])
            v_conf = float(row["video_confidence"])
        except ValueError:
            print("Invalid numeric row detected:", row)
            continue

        f_res = AMCEF.fuse(a_pred, a_conf, v_pred, v_conf)

        f_pred = f_res["fused_emotion"]
        eri_score = f_res["eri_score"]
        eri_level = f_res["eri_level"]

        y_true.append(gt)
        y_audio.append(a_pred)
        y_video.append(v_pred)
        y_fusion.append(f_pred)

        eri_scores.append(eri_score)
        eri_total[eri_level] += 1
        if f_pred == gt:
            eri_correct[eri_level] += 1

        # Fusion usefulness
        if f_pred == gt and (a_pred != gt or v_pred != gt):
            fusion_helped += 1
        if f_pred != gt and (a_pred == gt or v_pred == gt):
            fusion_hurt += 1

        confusion.setdefault((gt, f_pred), 0)
        confusion[(gt, f_pred)] += 1

        # Save enriched row
        row["fused_prediction"] = f_pred
        row["eri_score"] = eri_score
        row["eri_level"] = eri_level
        row["fusion_source"] = f_res["source"]
        enriched_rows.append(row)

    # --------------------------------------------------
    # Accuracy Metrics
    # --------------------------------------------------
    acc_audio = calculate_accuracy(y_true, y_audio)
    acc_video = calculate_accuracy(y_true, y_video)
    acc_fusion = calculate_accuracy(y_true, y_fusion)

    print("\nOverall Accuracy")
    print(f"Audio  : {acc_audio:.4f}")
    print(f"Video  : {acc_video:.4f}")
    print(f"Fusion : {acc_fusion:.4f}")

    # --------------------------------------------------
    # Non-neutral accuracy
    # --------------------------------------------------
    nn_idx = [i for i, y in enumerate(y_true) if y != "neutral"]
    nn_true = [y_true[i] for i in nn_idx]
    nn_fusion = [y_fusion[i] for i in nn_idx]

    nn_acc = calculate_accuracy(nn_true, nn_fusion)

    print("\nNon-Neutral Accuracy (Critical Metric)")
    print(f"Fusion (Non-Neutral): {nn_acc:.4f}")

    # --------------------------------------------------
    # Fusion behavior
    # --------------------------------------------------
    print("\nFusion Behavior Analysis")
    print(f"Fusion helped correct mistakes : {fusion_helped}")
    print(f"Fusion overruled correct model : {fusion_hurt}")

    # --------------------------------------------------
    # ERI Analysis (KEY NOVELTY)
    # --------------------------------------------------
    print("\nEmotion Reliability Index (ERI) Analysis")

    for level in ["HIGH", "MODERATE", "LOW"]:
        if eri_total[level] == 0:
            continue
        acc = eri_correct[level] / eri_total[level]
        print(f"{level:<9} → Samples: {eri_total[level]:>4}, Accuracy: {acc:.4f}")

    avg_eri = sum(eri_scores) / len(eri_scores) if eri_scores else 0.0
    print(f"\nAverage ERI Score: {avg_eri:.3f}")

    # --------------------------------------------------
    # Confusion summary
    # --------------------------------------------------
    print("\nTop Confusion Patterns")
    sorted_conf = sorted(confusion.items(), key=lambda x: x[1], reverse=True)

    for (gt, pred), count in sorted_conf[:10]:
        if gt != pred:
            print(f"{gt:>10} → {pred:<10} : {count}")

    # --------------------------------------------------
    # Save detailed results
    # --------------------------------------------------
    output_path = "assets/evaluation_results.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=enriched_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(enriched_rows)

    print(f"\nDetailed evaluation saved to {output_path}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    evaluate()
