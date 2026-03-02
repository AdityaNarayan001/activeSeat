import sys; sys.path.insert(0, 'src')
import numpy as np
from activeseat.params import SeatParams
from activeseat.plant import build_state_space
from scipy.linalg import solve_continuous_are

p = SeatParams()
A, Bu, Bw, C, D, ss = build_state_space(p)
print('ms_eff =', p.effective_seat_mass)
print('A:')
print(A)
print('Bu:', Bu.flatten())
print('C_accel:', C.flatten())

configs = [
    ('Current', [5e5, 100, 1e6, 100], 1e-6),
    ('AccelWt', [0, 711, 444444, 711], 1e-4),
    ('Balanced', [1000, 5000, 200000, 20000], 1e-4),
    ('Comfort', [10, 1000, 100000, 10000], 1e-3),
]

x_test = np.array([0.01, 0.5, 0.005, 0.5])

for name, Qd, R in configs:
    Q = np.diag(Qd)
    Rm = np.atleast_2d(R)
    P = solve_continuous_are(A, Bu, Q, Rm)
    K = np.linalg.solve(Rm, Bu.T @ P)
    Fa = -float(K @ x_test)
    eigs = np.linalg.eigvals(A - Bu @ K)
    print('\n%s (R=%.1e):' % (name, R))
    print('  K_force =', K.flatten())
    print('  Test Fa = %.0f N (max = %.0f)' % (Fa, p.max_force))
    print('  CL eigs real:', np.sort(eigs.real))
