#ifndef VT_WEBSTER_FDTD_SOLVER_HPP
#define VT_WEBSTER_FDTD_SOLVER_HPP

#include <cstddef>
#include <vector>

#include "vt_geometry_simple.hpp"
#include "vt_solver_common.hpp"

namespace vt_simple {

struct TimeSignal {
    double sample_rate_hz = 0.0;
    std::vector<double> samples;
};

struct WebsterFDTDOptions {
    AcousticConstants constants;
    std::size_t spatial_node_count = 21;
    double cfl = 0.95;
    std::size_t observation_node = 0;
};

struct WebsterTimeResponse {
    double sample_rate_hz = 0.0;
    std::vector<double> x_grid_m;
    std::vector<double> area_grid_m2;
    std::vector<double> input_pressure;
    std::vector<double> output_pressure;
};

struct WebsterFieldTimeResponse {
    double sample_rate_hz = 0.0;
    std::vector<double> x_grid_m;
    std::vector<double> area_grid_m2;
    std::vector<double> input_pressure;
    std::vector<std::vector<double>> pressure_by_time;
};

TimeSignal make_linear_chirp_signal(double sample_rate_hz,
                                    double duration_s,
                                    double f0_hz,
                                    double f1_hz,
                                    double amplitude = 1.0);

WebsterTimeResponse solve_webster_fdtd(
    const AreaProfile& profile,
    const TimeSignal& input_signal,
    const WebsterFDTDOptions& options = WebsterFDTDOptions());

WebsterFieldTimeResponse solve_webster_fdtd_field(
    const AreaProfile& profile,
    const TimeSignal& input_signal,
    const WebsterFDTDOptions& options = WebsterFDTDOptions());

std::vector<FrequencySample> estimate_transfer_function_from_signals(
    const std::vector<double>& input_signal,
    const std::vector<double>& output_signal,
    double sample_rate_hz,
    const std::vector<double>& frequencies_hz);

}  // namespace vt_simple

#endif  // VT_WEBSTER_FDTD_SOLVER_HPP
