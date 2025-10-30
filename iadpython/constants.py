"""List of constants used in inverse adding-doubling."""
import numpy as np

AD_MAX_THICKNESS = 1e6

# Physical constants
hck = 1.4387752e4     # hc/k_B  [µm·K]
c1  = 1.191042e8      # 2hc²    [W·µm⁴·m⁻²·sr⁻¹]

def planck_irradiance(lam_um, T):
    """Planck radiance  [W m⁻² µm⁻¹ sr⁻¹]"""
    x = hck / (lam_um * T)
    return c1 / (lam_um**5 * (np.exp(x) - 1.0))

def Temp_from_plank(lam_um, I):
    """Brightness temp (inverse Planck)."""
    return hck / (lam_um * np.log(c1/(lam_um**5 * I) + 1.0))