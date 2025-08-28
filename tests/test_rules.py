from src.rules_engine import load_policy, label_one

def test_ads_detection():
    cfg = load_policy("configs/policy.yaml")
    labels, _, _ = label_one("Visit www.example.com for 50% discount!", "Cafe", "Restaurant", "Good food", cfg)
    assert labels["ads"] == 1

def test_rant_no_visit_detection():
    cfg = load_policy("configs/policy.yaml")
    labels, _, _ = label_one("I haven't been here but I heard it's bad", "Cafe", "Restaurant", "Good food", cfg)
    assert labels["rant_no_visit"] == 1

def test_irrelevant_similarity():
    cfg = load_policy("configs/policy.yaml")
    # Off-topic relative to a pizzeria
    labels, scores, _ = label_one("I love my new iPhone camera", "Mario's Pizzeria", "Restaurant, Italian", "Neapolitan pizza", cfg)
    assert labels["irrelevant"] == 1
