# Quantum_offline_bellman_q_regressor 



import random
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

random.seed(39)
np.random.seed(39)



FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90_loto_2981.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90_loto_plus_1737.csv"),
)



NUM_QUBITS = 6
STATE_DIM = 2 ** NUM_QUBITS
MAX_NUM = 39
NUM_POSITIONS = 7
WINDOW_LEN = 6
GAMMA_GRID = (0.85, 0.95)
REG_GRID = (1e-3, 1e-2)


def load_draws(file_path: Path) -> np.ndarray:
    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean:
                continue
            parts = [p.strip() for p in clean.split(",") if p.strip()]
            nums = [int(x) for x in parts[:NUM_POSITIONS]]
            if len(nums) != NUM_POSITIONS:
                raise ValueError(f"Neispravan broj elemenata u redu: {clean}")
            if len(set(nums)) != NUM_POSITIONS or any(n < 1 or n > MAX_NUM for n in nums):
                raise ValueError(f"Neispravni brojevi u redu: {clean}")
            rows.append(sorted(nums))
    data = np.array(rows, dtype=np.int64)
    if len(data) < WINDOW_LEN + 30:
        raise ValueError(f"Nedovoljno istorijskih kola u fajlu: {file_path}")
    return data


def build_state_transitions(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_total = len(draws)
    s_curr = []
    s_next = []
    y_action = []
    for t in range(WINDOW_LEN, n_total):
        curr_win = draws[t - WINDOW_LEN : t].astype(np.float64) / float(MAX_NUM)
        next_win = draws[t - WINDOW_LEN + 1 : t + 1].astype(np.float64) / float(MAX_NUM)
        s_curr.append(curr_win.reshape(-1))
        s_next.append(next_win.reshape(-1))
        y_action.append(draws.__getitem__(t))
    return (
        np.array(s_curr, dtype=np.float64),
        np.array(s_next, dtype=np.float64),
        np.array(y_action, dtype=np.int64),
    )


def build_quantum_state_unitary(state_vec: np.ndarray) -> np.ndarray:
    qc = QuantumCircuit(NUM_QUBITS)
    n_dim = len(state_vec)
    for q in range(NUM_QUBITS):
        s_idx = (q * NUM_POSITIONS) % n_dim
        e_idx = s_idx + NUM_POSITIONS
        seg = state_vec[s_idx:e_idx]
        mean_val = float(np.mean(seg))
        span_val = float(seg.__getitem__(NUM_POSITIONS - 1) - seg.__getitem__(0))
        mid_val = float(seg.__getitem__(3))
        qc.ry(np.pi * (mean_val - 0.5), q)
        qc.rz(np.pi * span_val, q)
        qc.rx(0.5 * np.pi * (mid_val - 0.5), q)
    for q in range(NUM_QUBITS - 1):
        qc.cx(q, q + 1)
    qc.cx(NUM_QUBITS - 1, 0)
    return Operator(qc).data


def build_policy_ansatz_unitary(theta: np.ndarray) -> np.ndarray:
    qc = QuantumCircuit(NUM_QUBITS)
    idx = 0
    for layer in range(2):
        for q in range(NUM_QUBITS):
            qc.ry(float(theta.__getitem__(idx)), q)
            idx += 1
            qc.rz(float(theta.__getitem__(idx)), q)
            idx += 1
        for q in range(NUM_QUBITS - 1):
            qc.rxx(float(theta.__getitem__(idx)), q, q + 1)
            idx += 1
        qc.rxx(float(theta.__getitem__(idx)), NUM_QUBITS - 1, 0)
        idx += 1
        if layer == 0:
            for q in range(NUM_QUBITS - 1):
                qc.cz(q, q + 1)
    return Operator(qc).data


def encode_quantum_states(states_mat: np.ndarray) -> np.ndarray:
    n_samples = int(states_mat.shape.__getitem__(0))
    init_qc = QuantumCircuit(NUM_QUBITS)
    for q in range(NUM_QUBITS):
        init_qc.h(q)
    base_state = Statevector.from_int(0, dims=STATE_DIM).evolve(init_qc).data
    encoded = np.empty((n_samples, STATE_DIM), dtype=np.complex128)
    for i in range(n_samples):
        u_s = build_quantum_state_unitary(states_mat.__getitem__(i))
        encoded[i] = u_s @ base_state
    return encoded


def get_position_mask() -> np.ndarray:
    num_grid = np.arange(1, MAX_NUM + 1, dtype=np.int64)[None, :]
    pos_idx = np.arange(NUM_POSITIONS, dtype=np.int64)[:, None]
    min_allowed = pos_idx + 1
    max_allowed = MAX_NUM - (NUM_POSITIONS - 1 - pos_idx)
    return (num_grid >= min_allowed) & (num_grid <= max_allowed)


def compute_q_and_policy(
    q_states: np.ndarray,
    w_q: np.ndarray,
    b_q: np.ndarray,
    pos_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    amp_feats = np.concatenate([np.real(q_states), np.imag(q_states)], axis=1)
    q_vals = np.einsum("bf,pfk->bpk", amp_feats, w_q) + b_q[None, :, :]
    q_masked = np.where(pos_mask[None, :, :], q_vals, -1e9)
    q_shifted = q_masked - np.max(q_masked, axis=2, keepdims=True)
    exp_q = np.where(pos_mask[None, :, :], np.exp(np.clip(q_shifted, -30.0, 30.0)), 0.0)
    denom = np.sum(exp_q, axis=2, keepdims=True)
    uniform_fb = pos_mask[None, :, :].astype(np.float64) / np.sum(pos_mask, axis=1, keepdims=True)[None, :, :]
    policy_probs = np.where(
        denom > 1e-15,
        exp_q / np.maximum(denom, 1e-15),
        uniform_fb,
    )
    return q_vals, policy_probs, amp_feats


def fit_quantum_bellman_model(
    enc_curr: np.ndarray,
    enc_next: np.ndarray,
    y_train: np.ndarray,
    gamma: float,
    reg_weight: float,
    maxiter: int = 35,
) -> dict:
    rng = np.random.default_rng(39)
    n_theta = 2 * (3 * NUM_QUBITS)
    n_amp_feats = 2 * STATE_DIM
    w_shape = (NUM_POSITIONS, n_amp_feats, MAX_NUM)
    b_shape = (NUM_POSITIONS, MAX_NUM)

    n_w = int(np.prod(w_shape))
    n_b = int(np.prod(b_shape))

    init_theta = rng.uniform(-0.15, 0.15, size=n_theta)
    init_w = rng.normal(0.0, 0.03, size=n_w)
    init_b = np.zeros(n_b, dtype=np.float64)
    init_params = np.concatenate([init_theta, init_w, init_b])

    pos_mask = get_position_mask()
    n_samples = len(y_train)
    sample_idx = np.arange(n_samples)[:, None]
    pos_idx = np.arange(NUM_POSITIONS)[None, :]
    target_idx = y_train - 1

    one_hot_a = np.zeros((n_samples, NUM_POSITIONS, MAX_NUM), dtype=np.float64)
    one_hot_a[sample_idx, pos_idx, target_idx] = 1.0
    n_total_params = float(len(init_params))
    eps = 1e-4

    def eval_objective(params: np.ndarray) -> float:
        theta = params[:n_theta]
        w_q = params[n_theta : n_theta + n_w].reshape(w_shape)
        b_q = params[n_theta + n_w :].reshape(b_shape)

        u_pi = build_policy_ansatz_unitary(theta)
        psi_curr = enc_curr @ u_pi.T
        psi_next = enc_next @ u_pi.T

        q_curr, pi_curr, _ = compute_q_and_policy(psi_curr, w_q, b_q, pos_mask)
        q_next, pi_next, _ = compute_q_and_policy(psi_next, w_q, b_q, pos_mask)

        chosen_pi = pi_curr[sample_idx, pos_idx, target_idx]
        policy_nll = -float(np.mean(np.sum(np.log(np.maximum(chosen_pi, 1e-12)), axis=1)))

        q_sa = q_curr[sample_idx, pos_idx, target_idx]
        v_next = np.sum(pi_next * q_next, axis=2)
        reward_sa = np.log(np.maximum(chosen_pi, 1e-12))
        bellman_target = reward_sa + gamma * v_next
        td_loss = 0.5 * float(np.mean(np.sum((q_sa - bellman_target) ** 2, axis=1)))

        reg_loss = reg_weight * float(np.mean(params ** 2))
        return policy_nll + 0.25 * td_loss + reg_loss

    def loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
        theta = params[:n_theta]
        w_q = params[n_theta : n_theta + n_w].reshape(w_shape)
        b_q = params[n_theta + n_w :].reshape(b_shape)

        u_pi = build_policy_ansatz_unitary(theta)
        psi_curr = enc_curr @ u_pi.T
        psi_next = enc_next @ u_pi.T

        q_curr, pi_curr, feat_curr = compute_q_and_policy(psi_curr, w_q, b_q, pos_mask)
        q_next, pi_next, feat_next = compute_q_and_policy(psi_next, w_q, b_q, pos_mask)

        chosen_pi = pi_curr[sample_idx, pos_idx, target_idx]
        policy_nll = -float(np.mean(np.sum(np.log(np.maximum(chosen_pi, 1e-12)), axis=1)))

        q_sa = q_curr[sample_idx, pos_idx, target_idx]
        v_next = np.sum(pi_next * q_next, axis=2)
        reward_sa = np.log(np.maximum(chosen_pi, 1e-12))
        bellman_target = reward_sa + gamma * v_next
        td_err = q_sa - bellman_target
        td_loss = 0.5 * float(np.mean(np.sum(td_err ** 2, axis=1)))

        reg_loss = reg_weight * float(np.mean(params ** 2))
        total_loss = policy_nll + 0.25 * td_loss + reg_loss

        grad_logits_curr = (pi_curr - one_hot_a) * pos_mask[None, :, :] / float(n_samples)
        td_scale = 0.25 / float(n_samples)
        grad_q_curr = np.zeros_like(q_curr)
        grad_q_curr[sample_idx, pos_idx, target_idx] = td_scale * td_err

        grad_q_next = -(td_scale * gamma * td_err)[:, :, None] * pi_next * pos_mask[None, :, :]

        total_g_curr = grad_logits_curr + grad_q_curr
        grad_w = (
            np.einsum("bf,bpk->pfk", feat_curr, total_g_curr)
            + np.einsum("bf,bpk->pfk", feat_next, grad_q_next)
            + (2.0 * reg_weight / n_total_params) * w_q
        )
        grad_b = (
            np.sum(total_g_curr, axis=0)
            + np.sum(grad_q_next, axis=0)
            + (2.0 * reg_weight / n_total_params) * b_q
        )

        grad_theta = np.zeros(n_theta, dtype=np.float64)
        for t_idx in range(n_theta):
            p_plus = params.copy()
            p_plus[t_idx] += eps
            grad_theta[t_idx] = (eval_objective(p_plus) - total_loss) / eps

        grad = np.concatenate([grad_theta, grad_w.reshape(-1), grad_b.reshape(-1)])
        return total_loss, grad









    res = minimize(
        loss_and_grad,
        init_params,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "gtol": 1e-3},
    )









    opt = res.x
    theta_opt = opt[:n_theta]
    return {
        "u_pi": build_policy_ansatz_unitary(theta_opt),
        "w_q": opt[n_theta : n_theta + n_w].reshape(w_shape),
        "b_q": opt[n_theta + n_w :].reshape(b_shape),
        "pos_mask": pos_mask,
    }




def evaluate_policy_nll(model: dict, enc_states: np.ndarray, y_true: np.ndarray) -> float:
    psi = enc_states @ model.__getitem__("u_pi").T
    _, pi_probs, _ = compute_q_and_policy(
        psi,
        model.__getitem__("w_q"),
        model.__getitem__("b_q"),
        model.__getitem__("pos_mask"),
    )
    n_samples = len(y_true)
    sample_idx = np.arange(n_samples)[:, None]
    pos_idx = np.arange(NUM_POSITIONS)[None, :]
    target_idx = y_true - 1
    chosen = pi_probs[sample_idx, pos_idx, target_idx]
    return -float(np.mean(np.sum(np.log(np.maximum(chosen, 1e-12)), axis=1)))


def decode_optimal_bellman_sequence(q_values: np.ndarray, policy_probs: np.ndarray) -> list[int]:
    combined_score = np.log(np.maximum(policy_probs, 1e-12)) + 0.15 * q_values
    best_total = -1e18
    best_seq: list[int] = []

    def search(pos: int, prev_num: int, acc: float, seq: list[int]) -> None:
        nonlocal best_total, best_seq
        if pos == NUM_POSITIONS:
            if acc > best_total:
                best_total = acc
                best_seq = list(seq)
            return

        remaining = NUM_POSITIONS - 1 - pos
        low = prev_num + 1
        high = MAX_NUM - remaining
        row_s = combined_score.__getitem__(pos)
        row_p = policy_probs.__getitem__(pos)

        cands = []
        for num in range(low, high + 1):
            if float(row_p.__getitem__(num - 1)) > 0.0:
                cands.append((float(row_s.__getitem__(num - 1)), num))
        cands.sort(key=lambda item: item.__getitem__(0), reverse=True)

        for val, num in cands[:5]:
            if acc + val <= best_total:
                break
            seq.append(num)
            search(pos + 1, num, acc + val, seq)
            seq.pop()

    search(0, 0, 0.0, [])
    if len(best_seq) != NUM_POSITIONS:
        raise RuntimeError("Neuspesno dekodiranje 7 razlicitih brojeva.")
    return best_seq


def predict_next_draw(model: dict, draws: np.ndarray) -> list[int]:
    latest_win = (draws[-WINDOW_LEN:].astype(np.float64) / float(MAX_NUM)).reshape(1, -1)
    enc_latest = encode_quantum_states(latest_win)
    psi_latest = enc_latest @ model.__getitem__("u_pi").T
    q_vals, pi_probs, _ = compute_q_and_policy(
        psi_latest,
        model.__getitem__("w_q"),
        model.__getitem__("b_q"),
        model.__getitem__("pos_mask"),
    )
    return decode_optimal_bellman_sequence(q_vals.__getitem__(0), pi_probs.__getitem__(0))


def main() -> None:
    print("Metoda: Quantum Offline Bellman Q-Distribution Regressor (1x6 qubita)")
    print("Posebnost: Deterministicka Bellman-ova jednacina i politika nad kvantnim amplitudama na simulatoru")

    for file_path in FILES:
        if not file_path.exists():
            print(f"Fajl nije pronadjen: {file_path}")
            continue

        draws = load_draws(file_path)
        s_curr, s_next, y_all = build_state_transitions(draws)
        enc_curr_all = encode_quantum_states(s_curr)
        enc_next_all = encode_quantum_states(s_next)

        n_samples = int(y_all.shape.__getitem__(0))
        train_end = int(0.75 * n_samples)
        val_end = int(0.88 * n_samples)

        enc_c_tr, enc_n_tr, y_tr = enc_curr_all[:train_end], enc_next_all[:train_end], y_all[:train_end]
        enc_c_va, y_va = enc_curr_all[train_end:val_end], y_all[train_end:val_end]
        enc_c_te, y_te = enc_curr_all[val_end:], y_all[val_end:]

        best_gamma = float(GAMMA_GRID.__getitem__(0))
        best_reg = float(REG_GRID.__getitem__(0))
        best_val_nll = float("inf")

        for gamma in GAMMA_GRID:
            for reg in REG_GRID:
                model = fit_quantum_bellman_model(
                    enc_c_tr,
                    enc_n_tr,
                    y_tr,
                    gamma=gamma,
                    reg_weight=reg,
                    maxiter=35,
                )
                val_nll = evaluate_policy_nll(model, enc_c_va, y_va)
                if val_nll < best_val_nll:
                    best_val_nll = val_nll
                    best_gamma = gamma
                    best_reg = reg

        eval_model = fit_quantum_bellman_model(
            enc_c_tr,
            enc_n_tr,
            y_tr,
            gamma=best_gamma,
            reg_weight=best_reg,
            maxiter=35,
        )
        test_nll = evaluate_policy_nll(eval_model, enc_c_te, y_te)

        final_model = fit_quantum_bellman_model(
            enc_curr_all,
            enc_next_all,
            y_all,
            gamma=best_gamma,
            reg_weight=best_reg,
            maxiter=35,
        )
        next_combo = predict_next_draw(final_model, draws)

        print(f"CSV: {file_path.name}")
        print(f"Validacija NLL (gamma={best_gamma}, reg={best_reg}): {best_val_nll:.4f}")
        print(f"Test NLL: {test_nll:.4f}")
        print(f"NEXT: {next_combo}")


if __name__ == "__main__":
    main()



"""
Metoda: Quantum Offline Bellman Q-Distribution Regressor (1x6 qubita)
Posebnost: Deterministicka Bellman-ova jednacina i politika nad kvantnim amplitudama na simulatoru

CSV: loto7_4718_k90.csv
Validacija NLL (gamma=0.95, reg=0.001): 21.1500
Test NLL: 21.0330
NEXT: [1, 8, 13, 23, 29, 33, 39]

CSV: loto7_4718_k90_loto_2981.csv
Validacija NLL (gamma=0.95, reg=0.001): 21.1684
Test NLL: 21.1314
NEXT: [1, 8, 13, 21, 24, 33, 39]

CSV: loto7_4718_k90_loto_plus_1737.csv
Validacija NLL (gamma=0.85, reg=0.001): 21.2899
Test NLL: 21.8491
NEXT: [3, 7, 17, 23, 28, 31, 39]
"""



"""
Deterministički Offline Quantum Value/Policy Regressor (Bellman-ov okvir na amplitudama).

	• U Loto 7/39 ne može da se menja okruženje akcijama, pa klasično nasumično istraživanje uvodi zabranjeni.
	• Bellman-ova jednačina i procena \(Q(s_t, a)\) po sekvencijalnim koracima (pozicijama 1-7) daju drugačiji matematički mehanizam od prethodnog Pipeline regresora: umesto statičke regresije iz jednog stanja, ovde vrednost izbora na poziciji k zavisi od očekivane kumulativne vrednosti preostalih pozicija \(k+1 \dots 7\) pomoću faktora γ.

Quantum Offline Bellman Q-Distribution Regressor (1x6 qubita) u Qiskit-u (rad sa amplitudama na lokalnom simulatoru), gde se Q-vrednosti i politika \(\pi(a\vert{}s_t)\) uče deterministički iz istorijskih prelaza uz Bellman-ov faktor umanjenja γ (bira se hronološkom validacijom zajedno sa regularizacijom).
"""
