"""
train_neural_ik.py

Trains the MLP-based neural inverse kinematics model described in:
  "Kinematics-Aware Diffusion Policy with Consistent 3D Observation and
   Action Space for Whole-Arm Robotic Manipulation" (arXiv:2512.17568)

Core idea (Section IV of the paper):
  The diffusion policy predicts actions as a set of N 3D nodes placed on
  the robot arm body (e.g. at each joint link). To execute these predicted
  node positions on the real robot, joint angles must be recovered.
  Rather than running a slow iterative IK solver at inference time, KADP
  trains an MLP to approximate the mapping:

      f : R^(N*3)  -->  R^(n_joints)
          node positions   joint angles

  Training data is generated purely from forward kinematics (FK):
    1. Sample random joint configurations q ~ Uniform(q_min, q_max)
    2. Run FK to obtain the world-frame 3D position of each selected node
    3. Train the MLP to invert this mapping: nodes --> q

  At inference time, predicted node positions from the diffusion policy
  are fed into this MLP to get a fast initial joint-angle estimate, which
  can optionally be refined by a Newton step.

  Key requirement from the paper's ablation:
    The number of nodes N must be sufficient to uniquely determine the
    full joint configuration. For a 7-DoF arm (e.g. Franka Panda) the
    paper uses 7 nodes (one per joint link). With only 3 nodes the
    configuration is under-determined and MLP-IK is not applicable.

Robot configuration (defaults match Franka Panda as used in the paper):
  - 7 revolute joints
  - 7 nodes: one at each joint link origin
  - DH / URDF parameters loaded via either:
      (a) pytorch_kinematics  (recommended, pip install pytorch-kinematics)
      (b) a simple analytical FK stub included below for quick testing

Usage:
  # Quick test with built-in analytic FK stub (no external deps needed):
  python train_neural_ik.py --use_stub_fk

  # With a real URDF via pytorch_kinematics:
  python train_neural_ik.py --urdf path/to/robot.urdf \
                             --end_links link1 link2 link3 link4 link5 link6 link7

  # Resume / fine-tune from a checkpoint:
  python train_neural_ik.py --checkpoint ik_mlp.pt --epochs 50
"""

import argparse
import os
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


# ---------------------------------------------------------------------------
# 1.  Forward kinematics helpers
# ---------------------------------------------------------------------------

def fk_pytorch_kinematics(joint_angles: torch.Tensor,
                           chain,
                           node_link_names: list[str]) -> torch.Tensor:
    """
    Compute world-frame 3D positions of selected links using pytorch_kinematics.

    Args:
        joint_angles : (B, n_joints)  joint angles in radians
        chain        : a pytorch_kinematics.SerialChain
        node_link_names : list of link names whose origins form the nodes

    Returns:
        nodes : (B, N, 3)  world-frame XYZ positions
    """
    import pytorch_kinematics as pk

    B = joint_angles.shape[0]
    # pytorch_kinematics expects (B, n_joints)
    transforms = chain.forward_kinematics(joint_angles, end_only=False)
    positions = []
    for name in node_link_names:
        tf = transforms[name]                   # Transform3d, batch size B
        # extract translation: (B, 3)
        mat = tf.get_matrix()                   # (B, 4, 4)
        positions.append(mat[:, :3, 3])         # (B, 3)
    return torch.stack(positions, dim=1)        # (B, N, 3)


def fk_stub(joint_angles: torch.Tensor) -> torch.Tensor:
    """
    Minimal planar-ish FK stub for a 7-DoF arm, for testing without a URDF.
    Link lengths alternate 0.3 m and 0.2 m along z then y axes.
    NOT physically accurate — only used to verify the training pipeline.

    Args:
        joint_angles : (B, 7)
    Returns:
        nodes : (B, 7, 3)
    """
    B = joint_angles.shape[0]
    device = joint_angles.device
    dtype = joint_angles.dtype

    L = [0.333, 0.0, 0.316, 0.0825, -0.0825, 0.0, 0.088]   # rough Panda link lengths
    nodes = []
    # Accumulate a simple revolute chain around alternating z/y axes
    pos = torch.zeros(B, 3, device=device, dtype=dtype)
    R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)

    axes = [
        torch.tensor([0., 0., 1.], device=device, dtype=dtype),
        torch.tensor([0., 1., 0.], device=device, dtype=dtype),
        torch.tensor([0., 0., 1.], device=device, dtype=dtype),
        torch.tensor([0., 1., 0.], device=device, dtype=dtype),  # neg
        torch.tensor([0., 0., 1.], device=device, dtype=dtype),
        torch.tensor([0., 1., 0.], device=device, dtype=dtype),
        torch.tensor([0., 0., 1.], device=device, dtype=dtype),
    ]
    offsets = [
        torch.tensor([0., 0., 0.333], device=device, dtype=dtype),
        torch.tensor([0., 0., 0.0],   device=device, dtype=dtype),
        torch.tensor([0., 0., 0.316], device=device, dtype=dtype),
        torch.tensor([0.0825, 0., 0.], device=device, dtype=dtype),
        torch.tensor([-0.0825, 0., 0.316], device=device, dtype=dtype),
        torch.tensor([0., 0., 0.],    device=device, dtype=dtype),
        torch.tensor([0.088, 0., 0.], device=device, dtype=dtype),
    ]

    def rot_axis_angle(axis, theta):
        """Rodrigues rotation, batched. axis: (3,), theta: (B,) -> (B,3,3)"""
        c = torch.cos(theta).unsqueeze(-1).unsqueeze(-1)   # (B,1,1)
        s = torch.sin(theta).unsqueeze(-1).unsqueeze(-1)
        ax = axis / (axis.norm() + 1e-8)
        K = torch.zeros(3, 3, device=device, dtype=dtype)
        K[0, 1] = -ax[2]; K[0, 2] =  ax[1]
        K[1, 0] =  ax[2]; K[1, 2] = -ax[0]
        K[2, 0] = -ax[1]; K[2, 1] =  ax[0]
        I = torch.eye(3, device=device, dtype=dtype)
        return I + s * K + (1 - c) * (K @ K)

    for i in range(7):
        theta = joint_angles[:, i]                            # (B,)
        Ri = rot_axis_angle(axes[i], theta)                   # (B,3,3)
        R = torch.bmm(R, Ri)
        off = offsets[i].unsqueeze(0).expand(B, -1)           # (B,3)
        pos = pos + torch.bmm(R, off.unsqueeze(-1)).squeeze(-1)
        nodes.append(pos.clone())

    return torch.stack(nodes, dim=1)   # (B, 7, 3)


# ---------------------------------------------------------------------------
# 2.  Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(n_samples: int,
                     joint_limits_low: torch.Tensor,
                     joint_limits_high: torch.Tensor,
                     fk_fn,
                     device: torch.device,
                     batch_size: int = 4096,
                     noise_std: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate (node_positions, joint_angles) pairs by random FK sampling.

    Args:
        n_samples         : total number of samples to generate
        joint_limits_low  : (n_joints,)  lower joint limits in radians
        joint_limits_high : (n_joints,)  upper joint limits in radians
        fk_fn             : callable(joint_angles: (B, n_joints)) -> (B, N, 3)
        device            : torch device
        batch_size        : how many samples to compute FK on at once
        noise_std         : optional Gaussian noise added to node positions
                            (data augmentation, increases robustness to FK error)

    Returns:
        X : (n_samples, N*3)  flattened node positions  [input to MLP]
        Y : (n_samples, n_joints)  joint angles          [target output]
    """
    n_joints = joint_limits_low.shape[0]
    X_list, Y_list = [], []
    generated = 0

    print(f"Generating {n_samples:,} FK samples ...")
    t0 = time.time()

    while generated < n_samples:
        bs = min(batch_size, n_samples - generated)
        # Uniform sample in joint space
        u = torch.rand(bs, n_joints, device=device)
        q = joint_limits_low + u * (joint_limits_high - joint_limits_low)  # (B, n_joints)

        with torch.no_grad():
            nodes = fk_fn(q)                          # (B, N, 3)

        if noise_std > 0.0:
            nodes = nodes + torch.randn_like(nodes) * noise_std

        X_list.append(nodes.reshape(bs, -1).cpu())   # (B, N*3)
        Y_list.append(q.cpu())
        generated += bs

        if generated % max(1, (n_samples // 10)) < batch_size:
            elapsed = time.time() - t0
            print(f"  {generated:>8,} / {n_samples:,}  ({elapsed:.1f}s)")

    X = torch.cat(X_list, dim=0)
    Y = torch.cat(Y_list, dim=0)
    print(f"Done. Dataset shape: X={tuple(X.shape)}, Y={tuple(Y.shape)}")
    return X, Y


# ---------------------------------------------------------------------------
# 3.  MLP architecture
# ---------------------------------------------------------------------------

class NeuralIKMLP(nn.Module):
    """
    MLP for neural inverse kinematics as described in KADP (arXiv:2512.17568).

    Maps flattened 3D node positions -> joint angles.

    Architecture follows standard practice for this class of problem:
      - Several fully-connected layers with ReLU activations
      - BatchNorm for training stability
      - No activation on the output layer (raw joint angle regression)

    Input  : (B, N*3)      flattened 3D positions of N arm nodes
    Output : (B, n_joints) predicted joint angles in radians
    """

    def __init__(self,
                 n_nodes: int,
                 n_joints: int,
                 hidden_dims: list[int] = (512, 512, 256, 256),
                 use_batchnorm: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        input_dim = n_nodes * 3
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, n_joints))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 4.  Training
# ---------------------------------------------------------------------------

def fk_error_metric(predicted_q: torch.Tensor,
                    target_nodes: torch.Tensor,
                    fk_fn) -> torch.Tensor:
    """
    Compute the mean Euclidean distance between FK(predicted_q) and target_nodes.
    This is the geometric IK error reported in the paper (in metres).

    Args:
        predicted_q  : (B, n_joints)
        target_nodes : (B, N, 3)
        fk_fn        : forward kinematics function
    Returns:
        scalar mean positional error
    """
    with torch.no_grad():
        pred_nodes = fk_fn(predicted_q)   # (B, N, 3)
        err = (pred_nodes - target_nodes).norm(dim=-1)   # (B, N)
        return err.mean()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Robot configuration
    # ------------------------------------------------------------------
    n_joints = args.n_joints
    n_nodes = args.n_nodes

    # Joint limits (radians) — Franka Panda defaults
    # Adjust these to match your robot.
    default_low  = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
    default_high = [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]
    q_low  = torch.tensor(default_low[:n_joints],  dtype=torch.float32, device=device)
    q_high = torch.tensor(default_high[:n_joints], dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # Forward kinematics function
    # ------------------------------------------------------------------
    if args.use_stub_fk:
        print("Using built-in FK stub (for testing only — not physically accurate).")
        fk_fn = lambda q: fk_stub(q.to(device))
    else:
        try:
            import pytorch_kinematics as pk
        except ImportError:
            raise ImportError(
                "pytorch_kinematics is required for URDF-based FK.\n"
                "Install with:  pip install pytorch-kinematics\n"
                "Or run with:   --use_stub_fk  for quick testing."
            )
        chain = pk.build_serial_chain_from_urdf(
            open(args.urdf).read(),
            end_link_name=args.end_links[-1]
        ).to(dtype=torch.float32, device=device)
        node_link_names = args.end_links
        fk_fn = lambda q: fk_pytorch_kinematics(q, chain, node_link_names)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    X, Y = generate_dataset(
        n_samples=args.n_samples,
        joint_limits_low=q_low,
        joint_limits_high=q_high,
        fk_fn=fk_fn,
        device=device,
        batch_size=args.gen_batch_size,
        noise_std=args.noise_std,
    )

    # Normalise inputs (node positions) to zero mean, unit std
    X_mean = X.mean(dim=0, keepdim=True)
    X_std  = X.std(dim=0, keepdim=True).clamp(min=1e-6)
    X_norm = (X - X_mean) / X_std

    # Normalise targets (joint angles) to [-1, 1] using known joint limits
    q_low_cpu  = q_low.cpu()
    q_high_cpu = q_high.cpu()
    Y_norm = 2.0 * (Y - q_low_cpu) / (q_high_cpu - q_low_cpu) - 1.0

    dataset = TensorDataset(X_norm, Y_norm, X.reshape(-1, n_nodes, 3))
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 4,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    hidden_dims = [int(h) for h in args.hidden_dims.split(",")]
    model = NeuralIKMLP(
        n_nodes=n_nodes,
        n_joints=n_joints,
        hidden_dims=hidden_dims,
        use_batchnorm=not args.no_batchnorm,
        dropout=args.dropout,
    ).to(device)
    print(f"\nModel:\n{model}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}\n")

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    # Cosine annealing with linear warm-up
    def lr_lambda(step):
        warmup = args.warmup_steps
        total  = args.epochs * len(train_loader)
        if step < warmup:
            return float(step) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_fk_err = float("inf")
    global_step = 0

    # Move normalisation tensors to device for FK error computation
    X_mean_dev = X_mean.to(device)
    X_std_dev  = X_std.to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        t_epoch = time.time()

        for x_batch, y_batch, nodes_batch in train_loader:
            x_batch    = x_batch.to(device)
            y_batch    = y_batch.to(device)
            nodes_batch = nodes_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x_batch)                 # (B, n_joints), normalised
            loss   = criterion(y_pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item() * x_batch.size(0)
            global_step += 1

        train_loss = train_loss_sum / train_size

        # ---- Validation ----
        model.eval()
        val_loss_sum = 0.0
        fk_err_sum   = 0.0
        val_count    = 0

        with torch.no_grad():
            for x_batch, y_batch, nodes_batch in val_loader:
                x_batch     = x_batch.to(device)
                y_batch     = y_batch.to(device)
                nodes_batch = nodes_batch.to(device)
                B = x_batch.size(0)

                y_pred = model(x_batch)
                val_loss_sum += criterion(y_pred, y_batch).item() * B

                # Denormalise predicted joint angles -> radians
                q_low_dev  = q_low.to(device)
                q_high_dev = q_high.to(device)
                q_pred = (y_pred + 1.0) * 0.5 * (q_high_dev - q_low_dev) + q_low_dev

                # Geometric FK error (metres) — the metric reported in the paper
                fk_err = fk_error_metric(q_pred, nodes_batch, fk_fn)
                fk_err_sum += fk_err.item() * B
                val_count  += B

        val_loss   = val_loss_sum / val_count
        val_fk_err = fk_err_sum   / val_count
        elapsed    = time.time() - t_epoch
        lr_now     = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:4d}/{args.epochs} | "
              f"train_loss={train_loss:.5f} | "
              f"val_loss={val_loss:.5f} | "
              f"val_fk_err={val_fk_err*100:.3f} cm | "
              f"lr={lr_now:.2e} | "
              f"{elapsed:.1f}s")

        # ---- Save checkpoint ----
        if val_fk_err < best_val_fk_err:
            best_val_fk_err = val_fk_err
            save_path = args.save_path
            torch.save({
                "epoch":       epoch,
                "model":       model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_fk_err":  val_fk_err,
                "X_mean":      X_mean,
                "X_std":       X_std,
                "q_low":       q_low.cpu(),
                "q_high":      q_high.cpu(),
                "n_nodes":     n_nodes,
                "n_joints":    n_joints,
                "hidden_dims": hidden_dims,
            }, save_path)
            print(f"  --> Saved best model (val_fk_err={val_fk_err*100:.3f} cm) to {save_path}")

    print(f"\nTraining complete. Best val FK error: {best_val_fk_err*100:.3f} cm")
    print(f"Model saved to: {args.save_path}")


# ---------------------------------------------------------------------------
# 5.  Inference helper (for use at KADP policy inference time)
# ---------------------------------------------------------------------------

def load_and_infer(checkpoint_path: str,
                   node_positions: np.ndarray,
                   device_str: str = "cpu") -> np.ndarray:
    """
    Load a trained neural IK model and predict joint angles from node positions.

    This is the function called by the KADP policy at inference time to convert
    diffusion-policy-predicted 3D node positions back to joint angles.

    Args:
        checkpoint_path : path to the .pt file saved during training
        node_positions  : (N, 3) or (B, N, 3) numpy array of node positions
        device_str      : "cpu" or "cuda"

    Returns:
        joint_angles : (n_joints,) or (B, n_joints) numpy array in radians
    """
    device = torch.device(device_str)
    ckpt   = torch.load(checkpoint_path, map_location=device)

    n_nodes  = ckpt["n_nodes"]
    n_joints = ckpt["n_joints"]
    hidden_dims = ckpt["hidden_dims"]

    model = NeuralIKMLP(n_nodes=n_nodes, n_joints=n_joints,
                        hidden_dims=hidden_dims).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    X_mean = ckpt["X_mean"].to(device)
    X_std  = ckpt["X_std"].to(device)
    q_low  = ckpt["q_low"].to(device)
    q_high = ckpt["q_high"].to(device)

    single = (node_positions.ndim == 2)
    if single:
        node_positions = node_positions[None]   # (1, N, 3)

    x = torch.tensor(node_positions, dtype=torch.float32, device=device)
    x = x.reshape(x.shape[0], -1)              # (B, N*3)
    x = (x - X_mean) / X_std                   # normalise

    with torch.no_grad():
        y_norm = model(x)                       # (B, n_joints), in [-1,1]

    # Denormalise to radians
    q = (y_norm + 1.0) * 0.5 * (q_high - q_low) + q_low

    result = q.cpu().numpy()
    return result[0] if single else result


# ---------------------------------------------------------------------------
# 6.  CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train neural IK MLP from KADP (arXiv:2512.17568)"
    )

    # Robot / FK
    p.add_argument("--use_stub_fk", action="store_true",
                   help="Use built-in FK stub instead of a real URDF (for testing).")
    p.add_argument("--urdf", type=str, default=None,
                   help="Path to robot URDF file (requires pytorch-kinematics).")
    p.add_argument("--end_links", nargs="+", default=None,
                   help="Ordered list of link names to use as nodes (one per joint).")
    p.add_argument("--n_joints", type=int, default=7,
                   help="Number of joints (default: 7, Franka Panda).")
    p.add_argument("--n_nodes", type=int, default=7,
                   help="Number of 3D nodes (must be >= n_joints for unique IK).")

    # Dataset
    p.add_argument("--n_samples", type=int, default=500_000,
                   help="Number of (nodes, q) pairs to generate via FK sampling.")
    p.add_argument("--gen_batch_size", type=int, default=4096,
                   help="Batch size for FK data generation.")
    p.add_argument("--noise_std", type=float, default=0.001,
                   help="Gaussian noise std added to node positions (metres) for augmentation.")
    p.add_argument("--val_split", type=float, default=0.1,
                   help="Fraction of data to use for validation.")

    # Model
    p.add_argument("--hidden_dims", type=str, default="512,512,256,256",
                   help="Comma-separated hidden layer sizes.")
    p.add_argument("--no_batchnorm", action="store_true",
                   help="Disable BatchNorm layers.")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Dropout probability (0 = disabled).")

    # Training
    p.add_argument("--epochs", type=int, default=100,
                   help="Number of training epochs.")
    p.add_argument("--batch_size", type=int, default=1024,
                   help="Training batch size.")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Peak learning rate.")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="AdamW weight decay.")
    p.add_argument("--warmup_steps", type=int, default=500,
                   help="Number of linear LR warm-up steps.")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader worker processes.")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU even if CUDA is available.")

    # I/O
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to checkpoint to resume from.")
    p.add_argument("--save_path", type=str, default="ik_mlp_best.pt",
                   help="Path to save the best model checkpoint.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.use_stub_fk and args.urdf is None:
        print("WARNING: neither --use_stub_fk nor --urdf specified.")
        print("Defaulting to --use_stub_fk for testing.\n")
        args.use_stub_fk = True

    if args.n_nodes < args.n_joints:
        print(f"WARNING: n_nodes ({args.n_nodes}) < n_joints ({args.n_joints}).")
        print("The paper (arXiv:2512.17568, ablation in Sec. V) shows that with")
        print("fewer nodes than joints the full configuration is under-determined")
        print("and MLP-based IK is not applicable. Consider increasing --n_nodes.")

    train(args)