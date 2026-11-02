"""Generate a wholly artificial example; no field/core data are used."""
import numpy as np
import pandas as pd


def generate(cores=80, seed=20261102):
    if cores < 10:
        raise ValueError("Use at least 10 synthetic cores.")
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(cores):
        phi, ps, pm = rng.uniform(.06, .18), rng.uniform(.05, .4), rng.uniform(.10, .45)
        rw = rng.uniform(.08, .12)
        m = 1.65 + .55 * ps + .35 * pm + .6 * ps * pm
        n = 1.65 + .6 * (pm - ps) ** 2 + .25 * pm
        r0 = rw * phi ** (-m) * np.exp(rng.normal(0, .01))
        for sw in (1.0, .85, .65, .45, .25):
            rt = r0 if sw == 1 else r0 * sw ** (-n) * np.exp(rng.normal(0, .02))
            rows.append(dict(core_id=f'SYNTHETIC_{i:04d}', phi=phi, p_small=ps,
                             p_medium=pm, Rw=rw, R0=r0, Rt=rt, Sw=sw,
                             synthetic_true_m=m, synthetic_true_n=n,
                             data_origin='SYNTHETIC'))
    return pd.DataFrame(rows)
