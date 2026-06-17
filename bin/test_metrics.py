
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from bin.classify_anomalies import _compute_metrics

def test_weighted_metrics():
    # Dummy confusion matrix
    # Class A: 10 samples, 8 correct, 2 misclassified as B
    # Class B: 100 samples, 90 correct, 10 misclassified as A
    
    labels = ["A", "B"]
    confusion = {
        "A": {"A": 8, "B": 2},
        "B": {"A": 10, "B": 90}
    }
    
    # Expected calculations:
    # Class A:
    #   Support: 10
    #   TP: 8
    #   FP: 10 (from B->A)
    #   FN: 2 (from A->B)
    #   Precision: 8 / (8+10) = 8/18 = 0.4444
    #   Recall: 8 / (8+2) = 8/10 = 0.8
    #   F1: 2*0.4444*0.8 / (0.4444+0.8) = 0.7111 / 1.2444 = 0.5714
    
    # Class B:
    #   Support: 100
    #   TP: 90
    #   FP: 2 (from A->B)
    #   FN: 10 (from B->A)
    #   Precision: 90 / (90+2) = 90/92 = 0.9783
    #   Recall: 90 / (90+10) = 90/100 = 0.9
    #   F1: 2*0.9783*0.9 / (0.9783+0.9) = 1.7609 / 1.8783 = 0.9375
    
    # Weighted Average:
    #   Total Support: 110
    #   Weighted Precision: (0.4444*10 + 0.9783*100) / 110 = (4.444 + 97.83) / 110 = 102.274 / 110 = 0.9298
    #   Weighted Recall: (0.8*10 + 0.9*100) / 110 = (8 + 90) / 110 = 98 / 110 = 0.8909
    #   Weighted F1: (0.5714*10 + 0.9375*100) / 110 = (5.714 + 93.75) / 110 = 99.464 / 110 = 0.9042
    
    metrics = _compute_metrics(confusion, labels)
    
    print("Computed Metrics:")
    print(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall: {metrics['weighted_recall']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    
    assert abs(metrics['weighted_precision'] - 0.9298) < 0.001
    assert abs(metrics['weighted_recall'] - 0.8909) < 0.001
    assert abs(metrics['weighted_f1'] - 0.9042) < 0.001
    
    print("Test Passed!")

if __name__ == "__main__":
    test_weighted_metrics()
