"""
ghg_core.py -- core routines for the revised GHG fuzzy-gated compositional autoencoder study.

Everything needed to reproduce the manuscript numbers lives here:
  * data loading and composition construction (non-overlapping CRF parts)
  * multiplicative zero replacement
  * pivot (Helmert-type) ILR transform, explicit basis, exact inverse
  * Fuzzy C-Means with multi-start distinct-point initialisation and validity indices
  * NumPy autoencoders (plain / concatenation / gated with several gate sources) with Adam
  * PCA and Random-Forest decoders
Only numpy, scipy, pandas and scikit-learn are required.
"""
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------------------------
PARTS_20 = ['E_1A1', 'E_1A2', 'E_1A3', 'E_1A4', 'E_1A5', 'E_1B',
            'I_2A', 'I_2B', 'I_2C', 'I_2D', 'I_2E', 'I_2F', 'I_2G', 'I_2H',
            'A_3A', 'A_3B', 'A_3D', 'A_3X', 'W_5', 'O_6']

# Coarse 8-part amalgamation used for the multi-source-gas analysis (CO2, CH4, N2O)
AMALG_8 = {'EnergyInd': ['E_1A1'], 'Manuf': ['E_1A2'], 'Transport': ['E_1A3'],
           'OtherComb': ['E_1A4', 'E_1A5'], 'Fugitive': ['E_1B'],
           'IPPU': ['I_2A', 'I_2B', 'I_2C', 'I_2D', 'I_2E', 'I_2F', 'I_2G', 'I_2H'],
           'Agriculture': ['A_3A', 'A_3B', 'A_3D', 'A_3X'], 'WasteOther': ['W_5', 'O_6']}

GAS_ORDER = ['CO2', 'CH4', 'N2O', 'HFC_CO2E', 'PFC_CO2E', 'SF6_CO2E', 'NF3_CO2E']


def load_inventory(path, gases=None):
    """Load the Eurostat env_air_gge 2023 extract (one row per country x gas).
    Missing cells (Eurostat flag 'm' / not reported) are treated as zero emissions.
    The EU27 aggregate and the 'HFC/PFC not-specified mix' rows are removed."""
    d = pd.read_csv(path)
    d = d[(d.geo != 'EU27_2020') & (d.gas != 'HFC_PFC_NSP_CO2E')].copy()
    if gases is not None:
        d = d[d.gas.isin(gases)].copy()
    d[PARTS_20] = d[PARTS_20].fillna(0.0)
    d = d.reset_index(drop=True)
    return d


def amalgamate(d, scheme):
    X = np.column_stack([d[cols].to_numpy().sum(axis=1) for cols in scheme.values()])
    return X, list(scheme.keys())


def closure(X):
    X = np.asarray(X, float)
    return X / X.sum(axis=1, keepdims=True)


def mult_replace(X, delta):
    """Multiplicative zero replacement (Martin-Fernandez et al., 2003) on closed data.
    zeros -> delta ; non-zeros scaled by (1 - sum of imputed deltas). Row sums stay 1 and the
    ratios between non-zero parts are preserved exactly."""
    P = closure(X)
    Z = P == 0
    nz = Z.sum(axis=1, keepdims=True)
    out = np.where(Z, delta, P * (1.0 - nz * delta))
    return out


# ----------------------------------------------------------------------------------------------
# 2. ILR (pivot / Helmert-type balances)
# ----------------------------------------------------------------------------------------------
def ilr_basis(D):
    """Return the D x (D-1) contrast matrix V with orthonormal columns, V^T 1 = 0, such that
    z = clr(x) V reproduces  z_j = sqrt(j/(j+1)) * ln( gm(x_1..x_j) / x_{j+1} ),  j=1..D-1."""
    V = np.zeros((D, D - 1))
    for j in range(1, D):
        c = np.sqrt(j / (j + 1.0))
        V[:j, j - 1] = c / j
        V[j, j - 1] = -c
    return V


def clr(X):
    L = np.log(X)
    return L - L.mean(axis=1, keepdims=True)


def ilr(X, V):
    return clr(X) @ V


def ilr_inv(Z, V):
    return closure(np.exp(Z @ V.T))


# ----------------------------------------------------------------------------------------------
# 3. Fuzzy C-Means
# ----------------------------------------------------------------------------------------------
def _memberships(Z, C_, m):
    d2 = ((Z[:, None, :] - C_[None, :, :]) ** 2).sum(-1)
    d2 = np.maximum(d2, 1e-12)
    inv = d2 ** (-1.0 / (m - 1.0))
    return inv / inv.sum(axis=1, keepdims=True), d2


def fcm(Z, C, m=2.0, n_init=15, tol=1e-6, max_iter=1000, seed=0):
    """Fuzzy C-Means; each restart is initialised at C *distinct* training observations.
    Returns (centres, U, J) for the restart with the lowest converged objective."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        idx = rng.choice(len(Z), size=C, replace=False)
        cen = Z[idx].copy()
        J_old = np.inf
        for _it in range(max_iter):
            U, d2 = _memberships(Z, cen, m)
            Um = U ** m
            cen = (Um.T @ Z) / Um.sum(axis=0)[:, None]
            U, d2 = _memberships(Z, cen, m)
            J = float(((U ** m) * d2).sum())
            if abs(J_old - J) < tol * max(1.0, abs(J)):
                break
            J_old = J
        if best is None or J < best[2]:
            best = (cen, U, J)
    return best


def fcm_predict(Z, cen, m=2.0):
    return _memberships(Z, cen, m)[0]


def fcm_indices(Z, U, cen, m=2.0):
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    n, C = U.shape
    PC = float((U ** 2).sum() / n)
    PE = float(-(U * np.log(np.clip(U, 1e-300, None))).sum() / n)
    MPC = float(1 - C / (C - 1) * (1 - PC))
    d2 = ((Z[:, None, :] - cen[None, :, :]) ** 2).sum(-1)
    cd = ((cen[:, None, :] - cen[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(cd, np.inf)
    XB = float(((U ** m) * d2).sum() / (n * cd.min()))
    lab = U.argmax(1)
    if len(np.unique(lab)) > 1:
        SS = float(silhouette_score(Z, lab))
        CHI = float(calinski_harabasz_score(Z, lab))
    else:
        SS, CHI = np.nan, np.nan
    assert 1.0 / C - 1e-9 <= PC <= 1 + 1e-9, 'PC outside [1/C,1] -- implementation error'
    assert -1e-9 <= PE <= np.log(C) + 1e-9
    return dict(C=C, PC=PC, PE=PE, MPC=MPC, XB=XB, SS=SS, CHI=CHI,
                n_hard_clusters=int(len(np.unique(lab))))


# ----------------------------------------------------------------------------------------------
# 4. Autoencoders in NumPy (manual backprop, Adam, early stopping)
# ----------------------------------------------------------------------------------------------
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(a):
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=1, keepdims=True)


class AE:
    """z -> tanh(H) [optionally gated] -> tanh(L) -> tanh(H) -> linear(d).

    gate:
      'none'    : plain autoencoder
      'concat'  : side vector s concatenated to the input
      'side'    : g = sigmoid(s Wg + bg) multiplies the first hidden layer; s supplied externally
                  (FCM memberships, hard one-hot clusters, permuted memberships, gas labels ...)
      'moe'     : s = softmax(z Wr + br) learned end-to-end (K routes) -> g = sigmoid(s Wg + bg)
      'nf'      : s = softmax(-||z-c_k||^2 / (2 sigma^2)) with learnable centres c_k and sigma
                  (Gaussian fuzzy memberships trained jointly; TSK/ANFIS-style antecedent) -> gate
    """

    def __init__(self, d, H=64, L=4, gate='none', s_dim=0, K=4, seed=0):
        self.d, self.H, self.L, self.gate, self.K = d, H, L, gate, K
        r = np.random.default_rng(seed)
        din = d + (s_dim if gate == 'concat' else 0)

        def glorot(a, b):
            return r.normal(0, np.sqrt(2.0 / (a + b)), size=(a, b))
        p = dict(W1=glorot(din, H), b1=np.zeros(H), W2=glorot(H, L), b2=np.zeros(L),
                 W3=glorot(L, H), b3=np.zeros(H), W4=glorot(H, d), b4=np.zeros(d))
        if gate == 'side':
            p['Wg'] = glorot(s_dim, H); p['bg'] = np.zeros(H)
        if gate == 'moe':
            p['Wr'] = glorot(d, K); p['br'] = np.zeros(K)
            p['Wg'] = glorot(K, H); p['bg'] = np.zeros(H)
        if gate == 'nf':
            p['Cn'] = np.zeros((K, d)); p['ls'] = np.zeros(1)
            p['Wg'] = glorot(K, H); p['bg'] = np.zeros(H)
        self.p = p
        self._nf_init = gate == 'nf'

    def n_params(self):
        return int(sum(v.size for v in self.p.values()))

    # --- forward ------------------------------------------------------------------------------
    def forward(self, Z, S=None):
        p, c = self.p, {}
        X = np.hstack([Z, S]) if self.gate == 'concat' else Z
        c['X'] = X
        t1 = np.tanh(X @ p['W1'] + p['b1']); c['t1'] = t1
        if self.gate in ('side', 'moe', 'nf'):
            if self.gate == 'side':
                s = S
            elif self.gate == 'moe':
                s = _softmax(Z @ p['Wr'] + p['br'])
            else:
                diff = Z[:, None, :] - p['Cn'][None, :, :]
                dist = (diff ** 2).sum(-1)
                inv2s2 = np.exp(-2 * p['ls'][0]) / 2.0
                e = -dist * inv2s2
                s = _softmax(e)
                c['diff'], c['e'] = diff, e
            c['s'] = s
            g = _sig(s @ p['Wg'] + p['bg']); c['g'] = g
            h1 = t1 * g
        else:
            h1 = t1
        c['h1'] = h1
        code = np.tanh(h1 @ p['W2'] + p['b2']); c['code'] = code
        h2 = np.tanh(code @ p['W3'] + p['b3']); c['h2'] = h2
        Y = h2 @ p['W4'] + p['b4']
        return Y, c

    def gates(self, Z, S=None):
        _, c = self.forward(Z, S)
        return c.get('g')

    # --- backward -----------------------------------------------------------------------------
    def grads(self, Z, S=None, wd=0.0):
        p = self.p
        Y, c = self.forward(Z, S)
        n = Z.shape[0]
        R = Y - Z
        loss = float((R ** 2).mean())
        g = {}
        dY = 2 * R / R.size
        g['W4'] = c['h2'].T @ dY; g['b4'] = dY.sum(0)
        da3 = (dY @ p['W4'].T) * (1 - c['h2'] ** 2)
        g['W3'] = c['code'].T @ da3; g['b3'] = da3.sum(0)
        da2 = (da3 @ p['W3'].T) * (1 - c['code'] ** 2)
        g['W2'] = c['h1'].T @ da2; g['b2'] = da2.sum(0)
        dh1 = da2 @ p['W2'].T
        if 'g' in c:
            dt1 = dh1 * c['g']
            dgate = dh1 * c['t1']
            dsg = dgate * c['g'] * (1 - c['g'])
            g['Wg'] = c['s'].T @ dsg; g['bg'] = dsg.sum(0)
            if self.gate in ('moe', 'nf'):
                ds = dsg @ p['Wg'].T
                s = c['s']
                dlogit = s * (ds - (ds * s).sum(1, keepdims=True))
                if self.gate == 'moe':
                    g['Wr'] = Z.T @ dlogit; g['br'] = dlogit.sum(0)
                else:
                    inv_s2 = np.exp(-2 * p['ls'][0])
                    # e = -||diff||^2/(2 sigma^2); de/dC_k = diff_k / sigma^2
                    g['Cn'] = (dlogit[:, :, None] * c['diff']).sum(0) * inv_s2
                    g['ls'] = np.array([(dlogit * (-2.0 * c['e'])).sum()])
        else:
            dt1 = dh1
        da1 = dt1 * (1 - c['t1'] ** 2)
        g['W1'] = c['X'].T @ da1; g['b1'] = da1.sum(0)
        if wd > 0:
            for k in g:
                if k.startswith('W'):
                    g[k] = g[k] + 2 * wd * p[k]
                    loss_w = 0  # weight-decay term added to gradient only
        return loss, g

    def loss(self, Z, S=None):
        Y, _ = self.forward(Z, S)
        return float(((Y - Z) ** 2).mean())

    # --- training -----------------------------------------------------------------------------
    def fit(self, Ztr, Str=None, Zva=None, Sva=None, lr=0.01, wd=1e-3, epochs=3000, patience=150,
            seed=0):
        if self._nf_init:
            r = np.random.default_rng(seed + 991)
            self.p['Cn'] = Ztr[r.choice(len(Ztr), self.K, replace=False)].copy()
            med = np.median(((Ztr[:, None] - Ztr[None]) ** 2).sum(-1))
            self.p['ls'] = np.array([0.5 * np.log(max(med, 1e-6) / 2.0)])
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v = {k: np.zeros_like(v) for k, v in self.p.items()}
        b1, b2, eps = 0.9, 0.999, 1e-8
        best, best_p, wait, best_ep = np.inf, None, 0, 0
        for ep in range(1, epochs + 1):
            _, g = self.grads(Ztr, Str, wd)
            for k in self.p:
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
                mh = m[k] / (1 - b1 ** ep); vh = v[k] / (1 - b2 ** ep)
                self.p[k] = self.p[k] - lr * mh / (np.sqrt(vh) + eps)
            if Zva is not None:
                vl = self.loss(Zva, Sva)
                if vl < best - 1e-7:
                    best, wait, best_ep = vl, 0, ep
                    best_p = {k: x.copy() for k, x in self.p.items()}
                else:
                    wait += 1
                    if wait >= patience:
                        break
        if best_p is not None:
            self.p = best_p
        self.best_epoch = best_ep
        return self


def grad_check(gate, seed=0):
    """Finite-difference check of the manual backprop (used once, reported in the log)."""
    r = np.random.default_rng(seed)
    Z = r.normal(size=(7, 5)); S = _softmax(r.normal(size=(7, 3)))
    net = AE(5, H=6, L=2, gate=gate, s_dim=3, K=3, seed=seed)
    if gate == 'nf':
        net.p['Cn'] = r.normal(size=(3, 5)); net.p['ls'] = np.array([0.3])
    _, g = net.grads(Z, S if gate in ('side', 'concat') else None, wd=0.0)
    worst = 0.0
    for k, P in net.p.items():
        it = np.nditer(P, flags=['multi_index'])
        for _ in range(min(P.size, 12)):
            idx = it.multi_index
            old = P[idx]
            P[idx] = old + 1e-6; lp = net.loss(Z, S if gate in ('side', 'concat') else None)
            P[idx] = old - 1e-6; lm = net.loss(Z, S if gate in ('side', 'concat') else None)
            P[idx] = old
            num = (lp - lm) / 2e-6
            worst = max(worst, abs(num - g[k][idx]) / max(1e-8, abs(num) + abs(g[k][idx])))
            it.iternext()
    return worst


def matched_hidden(d, L, target, s_dim=0):
    """Smallest H such that a plain AE has at least `target` parameters."""
    H = 1
    while True:
        n = (d + 1) * H + (H + 1) * L + (L + 1) * H + (H + 1) * d
        if n >= target:
            return H
        H += 1
