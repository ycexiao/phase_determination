from diffpy.srfit.fitbase import Profile
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


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
            weights = np.array([*weights, 1 - sum(weights)])
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


def preprocess_profiles(
    target_data, ref_datas, normalization_peak_x_range=None
):
    """Use Profile class in diffpy.cmi to reuse the data processing workflow."""

    def find_peak_by_x(x_array, y_array, x_range):
        x_min, x_max = x_range
        mask = (x_array >= x_min) & (x_array <= x_max)
        y_sub = y_array[mask]
        peaks, _ = find_peaks(y_sub)
        if len(peaks) == 0:
            raise ValueError(f"No peaks found in the range {x_range}")
        # multiple peaks are possible, choose the highest one
        peak_idx = peaks[np.argmax(y_sub[peaks])]
        y_peak = y_sub[peak_idx]
        return y_peak

    if normalization_peak_x_range is not None:
        target_y = find_peak_by_x(
            target_data[0], target_data[1], normalization_peak_x_range[0]
        )
        ref_ys = [
            find_peak_by_x(
                ref_data[0],
                ref_data[1],
                normalization_peak_x_range[i + 1],
            )
            for i, ref_data in enumerate(ref_datas)
        ]
        target_data = (
            target_data[0],  # x values remain the same
            target_data[1] / target_y,  # normalize y values by the target peak
            target_data[2]
            / target_y,  # normalize uncertainty by the target peak
        )
        ref_datas = [
            (
                ref_data[0],
                ref_data[1] / ref_y,
                ref_data[2] / ref_y,
            )
            for ref_data, ref_y in zip(ref_datas, ref_ys)
        ]
    target_profile = Profile()
    target_profile.setObservedProfile(*target_data)
    ref_profiles = []
    for ref_data in ref_datas:
        ref_profile = Profile()
        ref_profile.setObservedProfile(*ref_data)
        ref_profile.setCalculationPoints(target_profile.xobs)
        ref_profiles.append(ref_profile)
    return target_profile, ref_profiles


def determine_phase_composition(
    target_profile,
    ref_profiles,
    mode="relaxed",
    initial_guess=None,
):
    """
    Parameters
    ----------
    target_profile: Profile
        The target phase profile.
    ref_profiles: list of Profile
        The reference phase profiles.
        ...
    """
    residual = residual_factory(target_profile, ref_profiles, mode=mode)
    if not initial_guess:
        if mode == "relaxed":
            initial_guess = [
                1 / len(ref_profiles) for i in range(len(ref_profiles))
            ]
        elif mode == "constrained":
            initial_guess = [
                1 / (len(ref_profiles) + 1)
                for i in range(len(ref_profiles) - 1)
            ]
    res = least_squares(residual, x0=initial_guess, bounds=(0, 1))
    return res


def post_fit_analysis(res, mode="relaxed"):
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

    if mode == "relaxed":
        for i, param in enumerate(res.x):
            print(f"Phase {i}: {param * 100:.2f}% ± {sigma[i] * 100:.2f}%")
        xs = res.x
    elif mode == "constrained":
        for i, param in enumerate(res.x):
            print(f"Phase {i}: {param * 100:.2f}% ± {sigma[i] * 100:.2f}%")
        last_x = 1 - sum(res.x)
        last_sigma = np.sqrt(np.sum(sigma**2))
        print(
            f"Phase {len(res.x)}: {last_x * 100:.2f}% ± {last_sigma * 100:.2f}%"
        )
        xs = [*res.x, last_x]
    print("chi2:", chi2)
    print("reduced chi2:", s2)
    print("Covariance matrix:\n", cov)
    print("Parameter uncertainties:", sigma)
    return xs


if __name__ == "__main__":
    import pandas as pd

    def read_data(file_path: str):
        data = pd.read_csv(
            file_path, sep=r"\s+", comment="#", header=None, names=["r", "G"]
        )
        x_array = data["r"].values
        y_array = data["G"].values
        uncertainty_array = np.ones_like(
            y_array
        )  # replace with actual uncertainty if available
        return x_array, y_array, uncertainty_array

    ref_region_1_file = "data/RefPosition1-Gr-Frame7555-map58-51-ROI58-51.csv"
    ref_region_2_file = "data/RefPosition2-Gr-Frame12406-map58-84-ROI58-84.csv"
    target_region_file = "data/Gr-Frame3145-map58-21-ROI58-21.csv"

    target_phase = read_data(target_region_file)
    ref_phase1 = read_data(ref_region_1_file)
    ref_phase2 = read_data(ref_region_2_file)

    ## Firstly, plot to get the peak positions for normalization
    # fig, axes = plt.subplots(1, 3, figsize=(8, 6))
    # axes[0].plot(target_phase[0], target_phase[1], label="Target Phase")
    # axes[1].plot(ref_phase1[0], ref_phase1[1], label="Reference Phase 1")
    # axes[2].plot(ref_phase2[0], ref_phase2[1], label="Reference Phase 2")
    # for ax in axes:
    #     ax.legend()
    # plt.show()

    target_profile, ref_profiles = preprocess_profiles(
        target_phase, [ref_phase1, ref_phase2], [(1, 2), (1, 2), (1, 2)]
    )
    mode = "constrained"
    # mode = "relaxed"
    res = determine_phase_composition(target_profile, ref_profiles, mode=mode)
    xs = post_fit_analysis(res, mode=mode)

    # validation
    fig, ax = plt.subplots()
    ax.plot(target_profile.xobs, target_profile.yobs, label="Target Profile")
    composed_profile = np.zeros_like(target_profile.yobs)
    for i, ref_profile in enumerate(ref_profiles):
        composed_profile += xs[i] * ref_profile.y
    ax.plot(target_profile.xobs, composed_profile, label="Composed Profile")
    ax.legend()
    plt.show()
