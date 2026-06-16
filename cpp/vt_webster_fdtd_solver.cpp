#include "vt_webster_fdtd_solver.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace vt_simple {
namespace {

using Complex = std::complex<double>;

double hann_value(std::size_t n, std::size_t size) {
    if (size < 2) {
        return 1.0;
    }
    return 0.5 * (1.0 - std::cos(2.0 * kPi * static_cast<double>(n) /
                                 static_cast<double>(size - 1)));
}

Complex dft_at_frequency(const std::vector<double>& signal,
                         double sample_rate_hz,
                         double frequency_hz) {
    if (signal.empty()) {
        return Complex(0.0, 0.0);
    }

    double mean = 0.0;
    for (double value : signal) {
        mean += value;
    }
    mean /= static_cast<double>(signal.size());

    Complex sum(0.0, 0.0);
    for (std::size_t n = 0; n < signal.size(); ++n) {
        const double t = static_cast<double>(n) / sample_rate_hz;
        const double window = hann_value(n, signal.size());
        const double x = (signal[n] - mean) * window;
        const double phase = -2.0 * kPi * frequency_hz * t;
        sum += x * std::exp(Complex(0.0, phase));
    }

    return sum;
}

}  // namespace

TimeSignal make_linear_chirp_signal(double sample_rate_hz,
                                    double duration_s,
                                    double f0_hz,
                                    double f1_hz,
                                    double amplitude) {
    if (!(sample_rate_hz > 0.0)) {
        throw std::invalid_argument("sample_rate_hz must be positive.");
    }
    if (!(duration_s > 0.0)) {
        throw std::invalid_argument("duration_s must be positive.");
    }
    if (!(f0_hz >= 0.0 && f1_hz > f0_hz)) {
        throw std::invalid_argument("Require 0 <= f0_hz < f1_hz.");
    }

    const std::size_t sample_count =
        static_cast<std::size_t>(std::ceil(duration_s * sample_rate_hz));

    TimeSignal signal;
    signal.sample_rate_hz = sample_rate_hz;
    signal.samples.resize(sample_count, 0.0);

    const double sweep_rate = (f1_hz - f0_hz) / duration_s;
    for (std::size_t n = 0; n < sample_count; ++n) {
        const double t = static_cast<double>(n) / sample_rate_hz;
        const double phase =
            2.0 * kPi * (f0_hz * t + 0.5 * sweep_rate * t * t);
        signal.samples[n] =
            amplitude * hann_value(n, sample_count) * std::sin(phase);
    }

    return signal;
}

WebsterFieldTimeResponse solve_webster_fdtd_field(
    const AreaProfile& profile,
    const TimeSignal& input_signal,
    const WebsterFDTDOptions& options) {
    validate_profile(profile);

    if (!(input_signal.sample_rate_hz > 0.0)) {
        throw std::invalid_argument(
            "input_signal.sample_rate_hz must be positive.");
    }
    if (input_signal.samples.empty()) {
        throw std::invalid_argument("input_signal.samples must not be empty.");
    }
    if (options.spatial_node_count < 3) {
        throw std::invalid_argument("spatial_node_count must be >= 3.");
    }
    if (!(options.cfl > 0.0 && options.cfl <= 1.0)) {
        throw std::invalid_argument("cfl must be in (0, 1].");
    }

    const double x_left = profile.points.front().x_m;
    const double x_right = profile.points.back().x_m;
    const double length_m = x_right - x_left;
    const std::size_t nx = options.spatial_node_count;
    const double dx = length_m / static_cast<double>(nx - 1);

    const double dt = 1.0 / input_signal.sample_rate_hz;
    const double courant = options.constants.c_m_s * dt / dx;
    if (courant > options.cfl + 1e-12) {
        throw std::invalid_argument(
            "Time step is too large for the chosen spatial grid: c*dt/dx exceeds cfl.");
    }

    std::vector<double> x_grid(nx, 0.0);
    std::vector<double> area_grid(nx, 0.0);
    for (std::size_t i = 0; i < nx; ++i) {
        x_grid[i] = x_left + static_cast<double>(i) * dx;
        area_grid[i] = area_at_linear(profile, x_grid[i]);
    }

    std::vector<double> area_half(nx - 1, 0.0);
    for (std::size_t i = 0; i + 1 < nx; ++i) {
        area_half[i] = 0.5 * (area_grid[i] + area_grid[i + 1]);
    }

    const std::size_t obs =
        (options.observation_node == 0)
            ? (nx - 2)
            : std::min(options.observation_node, nx - 2);

    std::vector<double> p_prev(nx, 0.0);
    std::vector<double> p_cur(nx, 0.0);
    std::vector<double> p_next(nx, 0.0);

    WebsterFieldTimeResponse response;
    response.sample_rate_hz = input_signal.sample_rate_hz;
    response.x_grid_m = x_grid;
    response.area_grid_m2 = area_grid;
    response.input_pressure = input_signal.samples;
    response.pressure_by_time.assign(
        input_signal.samples.size(),
        std::vector<double>(nx, 0.0));

    const double c2dt2 =
        options.constants.c_m_s * options.constants.c_m_s * dt * dt;
    const double inv_dx2 = 1.0 / (dx * dx);

    for (std::size_t n = 0; n < input_signal.samples.size(); ++n) {
        p_cur[0] = input_signal.samples[n];
        p_cur[nx - 1] = 0.0;

        for (std::size_t i = 1; i + 1 < nx; ++i) {
            const double flux_right =
                area_half[i] * (p_cur[i + 1] - p_cur[i]);
            const double flux_left =
                area_half[i - 1] * (p_cur[i] - p_cur[i - 1]);
            const double laplacian =
                (flux_right - flux_left) * inv_dx2 / area_grid[i];

            p_next[i] = 2.0 * p_cur[i] - p_prev[i] + c2dt2 * laplacian;
        }

        p_next[0] = input_signal.samples[n];
        p_next[nx - 1] = 0.0;

        response.pressure_by_time[n] = p_cur;

        p_prev.swap(p_cur);
        p_cur.swap(p_next);
        std::fill(p_next.begin(), p_next.end(), 0.0);
    }

    return response;
}

WebsterTimeResponse solve_webster_fdtd(const AreaProfile& profile,
                                       const TimeSignal& input_signal,
                                       const WebsterFDTDOptions& options) {
    const WebsterFieldTimeResponse field =
        solve_webster_fdtd_field(profile, input_signal, options);

    const std::size_t nx = field.x_grid_m.size();
    const std::size_t obs =
        (options.observation_node == 0)
            ? (nx - 2)
            : std::min(options.observation_node, nx - 2);

    WebsterTimeResponse response;
    response.sample_rate_hz = field.sample_rate_hz;
    response.x_grid_m = field.x_grid_m;
    response.area_grid_m2 = field.area_grid_m2;
    response.input_pressure = field.input_pressure;
    response.output_pressure.resize(field.pressure_by_time.size(), 0.0);

    for (std::size_t n = 0; n < field.pressure_by_time.size(); ++n) {
        response.output_pressure[n] = field.pressure_by_time[n][obs];
    }

    return response;
}

std::vector<FrequencySample> estimate_transfer_function_from_signals(
    const std::vector<double>& input_signal,
    const std::vector<double>& output_signal,
    double sample_rate_hz,
    const std::vector<double>& frequencies_hz) {
    if (!(sample_rate_hz > 0.0)) {
        throw std::invalid_argument("sample_rate_hz must be positive.");
    }
    if (input_signal.empty() || output_signal.empty()) {
        throw std::invalid_argument("Signals must not be empty.");
    }
    if (input_signal.size() != output_signal.size()) {
        throw std::invalid_argument(
            "Input and output signals must have the same length.");
    }

    std::vector<FrequencySample> spectrum;
    spectrum.reserve(frequencies_hz.size());

    for (double frequency_hz : frequencies_hz) {
        if (frequency_hz < 0.0) {
            throw std::invalid_argument("frequency_hz must be non-negative.");
        }

        const Complex input_fft =
            dft_at_frequency(input_signal, sample_rate_hz, frequency_hz);
        const Complex output_fft =
            dft_at_frequency(output_signal, sample_rate_hz, frequency_hz);

        FrequencySample sample;
        sample.frequency_hz = frequency_hz;
        sample.transfer =
            (std::abs(input_fft) < 1e-15)
                ? Complex(0.0, 0.0)
                : (output_fft / input_fft);
        spectrum.push_back(sample);
    }

    return spectrum;
}

}  // namespace vt_simple
