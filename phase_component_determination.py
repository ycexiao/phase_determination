import numpy
from scipy.optimize import least_squares
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import json


def least_squares_decomposition(
    target_pdf,
    reference_pdfs,
    mode,
    uncertainty_target_pdf=None,
    uncertainty_reference_pdfs=None,
    initial_guess=None,
):
    """Perform least squares decomposition of the target PDF using the reference PDFs.

    Parameters
    ----------
    target_pdf : numpy.ndarray
        The target probability density function to be decomposed.
    reference_pdfs : list of numpy.ndarray
        The reference probability density functions used for decomposition.
    mode : str, {"relaxed", "constrained"}
        The mode of decomposition to be used.
        relaxed mode:
            target_pdf = \sum a_i * reference_pdfs[i], a_i ≥ 0
        constrained mode
            target_pdf = \sum a_i * reference_pdfs[i], a_i ≥ 0, \sum a_i = 1
    initial_guess : list or numpy.ndarray, optional
        Initial guess for the decomposition coefficients.
    """

    def residual(a_s):
        calc_pdf = numpy.zeros_like(target_pdf)
        if mode == "constrained":
            a_s = [*a_s, 1 - sum(a_s)]
        elif mode == "relaxed":
            a_s = a_s
        else:
            raise ValueError("Invalid mode specified")
        calc_pdf = sum(a * ref_pdf for a, ref_pdf in zip(a_s, reference_pdfs))
        if (
            uncertainty_target_pdf is not None
            and uncertainty_reference_pdfs is not None
        ):
            sigma2 = uncertainty_target_pdf**2 + sum(
                (a * uncertainty_reference_pdfs[i]) ** 2
                for i, a in enumerate(a_s)
            )
            residuals = (calc_pdf - target_pdf) / numpy.sqrt(sigma2)
        else:
            residuals = calc_pdf - target_pdf
        return residuals

    if initial_guess is None:
        if mode == "relaxed":
            initial_guess = numpy.ones(len(reference_pdfs)) / len(
                reference_pdfs
            )
        elif mode == "constrained":
            initial_guess = numpy.ones(len(reference_pdfs) - 1) / len(
                reference_pdfs
            )
        else:
            raise ValueError("Invalid mode specified")
    res = least_squares(residual, initial_guess)
    return res


def preprocess_pdfs(pdfs, normalization_peak_range=None):
    """Normalize the target and reference PDFs."""

    def find_peak_by_x(x_array, y_array, x_range):
        x_min, x_max = x_range
        mask = (x_array >= x_min) & (x_array <= x_max)
        y_sub = y_array[mask]
        peaks, _ = find_peaks(y_sub)
        if len(peaks) == 0:
            raise ValueError(f"No peaks found in the range {x_range}")
        # multiple peaks are possible, choose the highest one
        peak_idx = peaks[numpy.argmax(y_sub[peaks])]
        y_peak = y_sub[peak_idx]
        return y_peak

    if normalization_peak_range is None:
        pdfs = [[pdf[0], pdf[1] / max(pdf[1])] for pdf in pdfs]
    else:
        max_intensity = max(
            [
                find_peak_by_x(pdf[0], pdf[1], normalization_peak_range)
                for pdf in pdfs
            ]
        )
        pdfs = [[pdf[0], pdf[1] / max_intensity] for pdf in pdfs]
    return pdfs


def result_analysis(scipy_result, mode):
    residual_vector = scipy_result.fun
    jacobian = scipy_result.jac
    chi2 = numpy.sum(residual_vector**2)
    reduced_chi2 = chi2 / (len(residual_vector) - len(scipy_result.x))
    cov = reduced_chi2 * numpy.linalg.inv(jacobian.T @ jacobian)
    sigma = numpy.sqrt(numpy.diag(cov))
    if mode == "relaxed":
        a_s = scipy_result.x
    elif mode == "constrained":
        last_a = 1 - sum(scipy_result.x)
        last_sigma = numpy.sqrt(numpy.sum(sigma**2))
        a_s = [*scipy_result.x, last_a]
        sigma = [*sigma, last_sigma]
    return {
        "a_s": list(a_s),
        "parameter_uncertainty": list(sigma),
        "chi2": chi2.tolist(),
        "reduced_chi2": reduced_chi2.tolist(),
        "covariance": cov.tolist(),
        "residual_vector": residual_vector.tolist(),
        "jacobian": jacobian.tolist(),
    }


def main(
    target_region,
    *reference_regions,
    uncertainty_target_region=None,
    uncertainty_reference_regions=None,
    initial_guess=None,
    normalization_peak_range=None,
    mode="relaxed",
    save_name="results.json",
):
    """Decompose the target_region PDF into the reference regions PDFs.

    Since xdata is not provided, the function assumes that the PDFs are
    already aligned. Only normalization is performed.

    Parameters
    ----------
    target_region : list
        region[x][y] = [r(A), G(au)]
    reference_regions : list
        List of reference regions PDFs.
    """
    reference_pdfs = [
        numpy.array(region).mean(axis=0).mean(axis=0)
        for region in reference_regions
    ]
    target_region = [
        preprocess_pdfs(x_slice, normalization_peak_range)
        for x_slice in target_region
    ]
    reference_pdfs = preprocess_pdfs(reference_pdfs, normalization_peak_range)
    reference_g = [pdf[1] for pdf in reference_pdfs]

    m = len(target_region)
    results = {
        "residual_vector": [[] for _ in range(m)],
        "jacobian": [[] for _ in range(m)],
        "reduced_chi2": [[] for _ in range(m)],
        "covariance": [[] for _ in range(m)],
        "parameter_uncertainty": [[] for _ in range(m)],
        "a_s": [[] for _ in range(m)],
    }
    for i, target_pdf_x_slice in enumerate(target_region):
        for j, target_pdf in enumerate(target_pdf_x_slice):
            target_g = target_pdf[1]
            res = least_squares_decomposition(
                target_g, reference_g, initial_guess=initial_guess, mode=mode
            )
            print(f"Completed ({i}, {j})")
            tmp_result = result_analysis(res, mode=mode)
            for key in results.keys():
                results[key][i].append(tmp_result[key])
    with open(save_name, "w") as f:
        json.dump(results, f, indent=2)
    return results


def pdf_visualization(target_pdf, reference_pdfs, a_s):
    reconstructed_pdf = sum(a * pdf[1] for a, pdf in zip(a_s, reference_pdfs))
    fig, ax = plt.subplots()
    ax.plot(target_pdf[0], target_pdf[1], label="Target PDF")
    ax.plot(target_pdf[0], reconstructed_pdf, label="Reconstructed PDF")
    ax.legend()
    plt.show()


def error_visualization(results, title=None):
    errors = results["residual_vector"]
    errors = numpy.array(errors)
    errors = [
        [numpy.sum(res_vector**2) for res_vector in res_vector_slice]
        for res_vector_slice in errors
    ]
    errors = numpy.array(errors)
    fig, ax = plt.subplots()
    im = ax.imshow(errors, vmin=0, vmax=1, cmap="viridis")
    if title is not None:
        ax.set_title(title)
    plt.colorbar(im)
    plt.show()