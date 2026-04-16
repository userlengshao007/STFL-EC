import argparse
import yaml
import torch
import random
import math
import numpy as np
import copy
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T

# Import the refactored modules
from models.mlp import MLP
from defense.embedding import vec, load_vec, embed_watermark
from defense.extraction import extract_watermark
from defense.tracing import find_leakage_cycle, dynamic_regroup, identify_traitor


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def local_train(model, dataloader, lr, device):
    """Perform one round of local training on the client side."""
    m = model.to(device)
    opt = optim.SGD(m.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    m.train()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        lossf(m(x), y).backward()
        opt.step()
    return m.cpu()


def main():
    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(description="STFL-EC Framework")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'trace'])
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--watermark_len', type=int, default=256)
    args = parser.parse_args()

    # 2. Load YAML configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Override YAML config with command-line arguments if provided
    if args.watermark_len:
        config['watermark_len'] = args.watermark_len

    set_seed()
    device = config.get('device', 'cpu')

    # ==========================================
    # MODE: TRAIN (Federated Learning & Watermarking)
    # ==========================================
    if args.mode == 'train':
        print(f"[*] Starting Training on {args.dataset.upper()} with Watermark Length: {config['watermark_len']}")

        # --- Prepare Data (Mocked for MNIST) ---
        tf = T.ToTensor()
        train_ds = torchvision.datasets.MNIST('./data', True, download=True, transform=tf)
        num_clients = config['num_clients']
        idx_split = np.array_split(np.random.permutation(len(train_ds)), num_clients)

        def make_loader(idxs):
            ds = torch.utils.data.Subset(train_ds, idxs)
            return torch.utils.data.DataLoader(ds, batch_size=config['batch_size'], shuffle=True)

        client_ld = [make_loader(i) for i in idx_split]

        # --- Initialize Global Model & Watermark Parameters ---
        global_m = MLP()
        param_dim = len(vec(global_m))
        wm_len = config['watermark_len']

        # Randomly assign watermark positions for each client
        perm = list(range(param_dim))
        random.shuffle(perm)
        wm_pos = [perm[i * wm_len: (i + 1) * wm_len] for i in range(num_clients)]

        # Generate target bit sequences for CYCLES x CLIENTS x ROUNDS
        wm_bits = {c: {cid: [torch.randint(0, 2, (wm_len,), dtype=torch.float32)
                             for _ in range(config['rounds_per_cycle'])]
                       for cid in range(num_clients)}
                   for c in range(config['cycles'])}

        # --- Federated Learning Loop ---
        for c in range(config['cycles']):
            print(f"\n--- Cycle {c + 1}/{config['cycles']} ---")
            # Select active client group
            num_selected = math.ceil(config['select_rate'] * num_clients)
            chosen_clients = random.sample(range(num_clients), k=num_selected)
            print(f"Selected Clients: {chosen_clients}")

            locals_ = []
            for cid in chosen_clients:
                local_m = copy.deepcopy(global_m)
                for r in range(config['rounds_per_cycle']):
                    # 1. Local Training
                    local_m = local_train(local_m, client_ld[cid], config['learning_rate'], device)
                    # 2. Watermark Embedding (EWE)
                    v = vec(local_m)
                    embed_watermark(v, wm_pos[cid], wm_bits[c][cid][r], config['alpha'])
                    load_vec(local_m, v)
                locals_.append(local_m)

            # --- FedAvg Aggregation ---
            with torch.no_grad():
                agg = copy.deepcopy(global_m.state_dict())
                for k in agg: agg[k].zero_()
                for lm in locals_:
                    for k, p in lm.state_dict().items():
                        agg[k] += p
                for k in agg: agg[k] /= len(locals_)
                global_m.load_state_dict(agg)

        print("\n[*] Training complete. Global model and watermarks successfully updated.")
        # torch.save(global_m.state_dict(), './checkpoints/global_model.pth')

    # ==========================================
    # MODE: TRACE (Traitor Identification)
    # ==========================================
    elif args.mode == 'trace':
        print("[*] Starting Traitor Tracing routine...")

        # ---------------------------------------------------------
        # Mocking the tracking process as described in the paper (Fig. 6)
        # ---------------------------------------------------------
        print("\n--- Phase 1: First Model Leakage Detected ---")
        # Historical records maintained by KMC
        group_history = {
            1: [11, 12, 13, 14, 15, 16],
            2: [1, 3, 5, 7, 8, 10],  # Group 2 from Cycle 3
            3: [21, 22, 23, 24, 25, 26]
        }

        # 1. Server audits Suspect Model 1, extracts watermark, matches cycle
        detected_cycle_1 = 3
        suspect_set_1 = group_history[detected_cycle_1 - 1]
        print(f"[*] Suspect model 1 matching highest score at cycle {detected_cycle_1}.")
        print(f"[*] Locking corresponding devices as Suspect Set 1: {suspect_set_1}")

        print("\n--- Phase 2: Dynamic Regrouping (Inter-group Reassignment) ---")
        # 2. Redistribute suspect clients among 6 new groups
        num_new_groups = 6
        new_groups = dynamic_regroup(suspect_set_1, num_new_groups)
        for g_id, members in new_groups.items():
            print(f"    - Group {g_id + 1} now contains suspect: {members}")

        print("\n--- Phase 3: Second Model Leakage Detected ---")
        # 3. Server audits Suspect Model 2, traces to Group 5 (containing device 8)
        detected_cycle_2 = 4
        suspect_set_2 = new_groups[4]  # Group 5 (index 4)
        print(f"[*] Suspect model 2 matching highest score at cycle {detected_cycle_2}.")
        print(f"[*] Locking newly identified suspect group: {suspect_set_2}")

        print("\n--- Phase 4: Identity Matching (Intersection) ---")
        # 4. Compute intersection to pinpoint the traitor
        traitors = identify_traitor(suspect_set_1, suspect_set_2)

        if len(traitors) == 1:
            print(f"[!] SUCCESS: The unique traitor is exactly identified as Device {traitors[0]}!")
        elif len(traitors) > 1:
            print(f"[!] Warning: Multiple colluding traitors detected (Case 3): {traitors}")
        else:
            print("[!] Tracing failed. No intersection found.")


if __name__ == "__main__":
    main()