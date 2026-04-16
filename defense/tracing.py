import torch


def calculate_match_score(extracted_wm, expected_wm):
    """
    Calculate the Match Score based on Normalized Hamming Distance (ND).
    Matches Equation (20) and (21) in the paper.
    """
    # Exact match ratio equals (1 - ND)
    return (extracted_wm == expected_wm).float().mean().item()


def find_leakage_cycle(extracted_wm, historical_wms):
    """
    Identify the training cycle in which the model leakage occurred.
    Matches Equation (22) in the paper: historical traversal matching strategy.

    Args:
        extracted_wm: The watermark sequence extracted from the suspicious model.
        historical_wms: A dict mapping {cycle_index: global_watermark}.
    Returns:
        best_cycle (int), max_score (float)
    """
    best_cycle = -1
    max_score = -1.0

    for cycle, hist_wm in historical_wms.items():
        score = calculate_match_score(extracted_wm, hist_wm)
        if score > max_score:
            max_score = score
            best_cycle = cycle

    return best_cycle, max_score


def dynamic_regroup(suspect_clients, num_groups):
    """
    Redistribute suspect clients among different groups.
    Ensures that (if possible) each group accommodates at most one suspicious device.
    """
    new_groups = {i: [] for i in range(num_groups)}
    for i, client_id in enumerate(suspect_clients):
        group_idx = i % num_groups
        new_groups[group_idx].append(client_id)

    return new_groups


def identify_traitor(suspect_set_round_1, suspect_set_round_2):
    """
    Uniquely identify the traitor by computing the intersection of
    two suspicious sets obtained from different leakage rounds.
    """
    traitors = list(set(suspect_set_round_1) & set(suspect_set_round_2))
    return traitors