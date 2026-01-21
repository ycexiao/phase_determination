from diffpy.srfit.fitbase import Profile
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


def residual_factory(
    target_profile,
    ref_profiles,
    mode="restricted",
):
    target_y = np.asarray(target_profile.yobs)
    uncertainty_target = np.asarray(target_profile.dyobs)
    ref_ys = np.asarray([ref_profile.y for ref_profile in ref_profiles])
    uncertianty_ref = np.asarray(
        [ref_profile.dy for ref_profile in ref_profiles]
    )

    def residual(weights):
        """
        Constrained:
            Equation:
                a_0*x_0 + a_1*x_1+... + (1-a_0-a_1-...)*xn
        Relaxed:
            Equation:
                a_0*x_0 + a_1*x_1+... + a_n*x_n
        """
        calc_y = np.zeros_like(target_y)
        if mode == "constrained":
            weights.append(1 - sum(weights))
        elif mode == "relaxed":
            pass
        else:
            raise ValueError(
                f"Unrecognized mode {mode}. "
                "Please choose between ['constrained' and 'relaxed']"
            )
        for weight, y in zip(weights, ref_ys):
            calc_y += weight * y
        if uncertianty_ref is not None and uncertainty_target is not None:
            sigma2 = uncertainty_target**2
            for weight, uncertainty in zip(weights, uncertianty_ref):
                sigma2 += weight**2 * uncertainty**2
            sigma = np.sqrt(sigma2)
        else:
            sigma = np.ones_like(target_y)
        return (target_y - calc_y) / sigma

    return residual


def convert_to_profile(x_array, y_array, uncertainty_array):
    """Use Profile class in diffpy.cmi to reuse the data processing workflow."""
    profile = Profile()
    profile.setObservedProfile(x_array, y_array, uncertainty_array)
    return profile


def determine_phase_composition(
    target_phase,
    ref_phases,
    mode="relaxed",
    initial_guess=None,
):
    """
    Parameters
    ----------
    target_phase: list
        The 2D array of target phase PDF.
        target_phase[0] = r, target_phase[1] = G, target_phase[2] = uncertainty
    ref_phases: list
        The 3D array of ref phase PDFs.
        ref_phase[0][0] = r_0, ref_phase[0][1] = G_0, ref_phase[0][2] = uncertainty
        ...
    """
    # preprocess the profiles
    target_profile = convert_to_profile(*target_phase)
    ref_profiles = [convert_to_profile(*ref_phase) for ref_phase in ref_phases]
    for p in ref_profiles:
        p.setCalculationPoints(target_profile.xobs)
    residual = residual_factory(target_profile, ref_profiles, mode=mode)
    if not initial_guess:
        if mode == "relaxed":
            initial_guess = [
                1 / len(ref_phases) for i in range(len(ref_phases))
            ]
        elif mode == "constrained":
            initial_guess = [
                1 / (len(ref_phases) + 1) for i in range(len(ref_phases))
            ]
    res = least_squares(residual, x0=initial_guess, bounds=(0, 1))
    return res


def post_fit_analysis(res):
    r = res.fun  # residual vector at solution
    J = res.jac  # Jacobian matrix at solution (shape len(r) x n_params)

    # 2. Degrees of freedom
    N = len(r)
    p = len(res.x)
    dof = N - p

    # 3. Estimate variance of residuals
    chi2 = np.sum(r**2)
    s2 = chi2 / dof  # reduced chi-squared

    # 4. Covariance matrix
    cov = s2 * np.linalg.inv(J.T @ J)

    # 5. Parameter uncertainties (1σ)
    sigma = np.sqrt(np.diag(cov))

    print("chi2:", chi2)
    print("reduced chi2:", s2)
    print("Covariance matrix:\n", cov)
    print("Parameter uncertainties:", sigma)


if __name__ == "__main__":
    import pandas as pd

    def read_data(file_path: str):
        data = pd.read_csv(
            file_path, sep=r"\s+", comment="#", header=None, names=["r", "G"]
        )
        x_array = data["r"].values
        y_array = data["G"].values
        return x_array, y_array, np.ones_like(y_array)

    ref_region_1_file = "data/RefPosition1-Gr-Frame7555-map58-51-ROI58-51.csv"
    ref_region_2_file = "data/RefPosition2-Gr-Frame12406-map58-84-ROI58-84.csv"
    target_region_file = "data/Gr-Frame3145-map58-21-ROI58-21.csv"

    target_phase = read_data(target_region_file)
    ref_phase1 = read_data(ref_region_1_file)
    ref_phase2 = read_data(ref_region_2_file)
    res = determine_phase_composition(target_phase, [ref_phase1, ref_phase2])
    print("Optimization result:", res)
    post_fit_analysis(res)

    # fig, axes = plt.subplots(1, 3, figsize=(8, 6))
    # axes[0].plot(target_phase[0], target_phase[1], label="Target Phase")
    # axes[1].plot(ref_phase1[0], ref_phase1[1], label="Reference Phase 1")
    # axes[2].plot(ref_phase2[0], ref_phase2[1], label="Reference Phase 2")
    # for ax in axes:
    #     ax.legend()
    # plt.show()
