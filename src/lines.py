# Minimal stub for smoke testing — project knowledge doesn't include lines.py
# Extended with attributes (nu0, f_lu, g_l, g_u) used by the MC kernel.

C_LIGHT_CMS = 2.99792458e10  # cm/s


class _Line:
    def __init__(self, lam0_cm, f_lu, g_l, g_u):
        self.lam0_cm = lam0_cm
        self.nu0 = C_LIGHT_CMS / lam0_cm
        self.f_lu = f_lu
        self.g_l = g_l
        self.g_u = g_u


LINE_LIB = {
    # H lines (NIST): statistical weights g = 2 n^2.
    'Halpha':   _Line(6562.80e-8, 0.6407, 8.0, 18.0),  # n=2 -> n=3
    'Hbeta':    _Line(4861.35e-8, 0.1193, 8.0, 32.0),  # n=2 -> n=4
    'Hgamma':   _Line(4340.47e-8, 0.0447, 8.0, 50.0),  # n=2 -> n=5
    'Hdelta':   _Line(4101.74e-8, 0.0221, 8.0, 72.0),  # n=2 -> n=6
    'HeI_5876': _Line(5875.62e-8, 0.6109, 9.0, 15.0),
    'HeII_4686':_Line(4685.71e-8, 0.8421, 18.0, 32.0),
}
