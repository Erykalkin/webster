#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt_arma_solver.hpp"
#include "vt_cone_reference_solver.hpp"
#include "vt_cylinder_tlm_solver.hpp"
#include "vt_geometry_simple.hpp"
#include "vt_webster_fdtd_solver.hpp"

namespace {

using vt_simple::AreaProfile;
using vt_simple::ProfilePoint;

std::string require_value(int& i, int argc, char** argv, const std::string& flag) {
    if (i + 1 >= argc) {
        throw std::invalid_argument("Missing value for " + flag);
    }
    ++i;
    return argv[i];
}

std::vector<double> parse_csv_doubles(const std::string& text) {
    std::vector<double> values;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            values.push_back(std::stod(token));
        }
    }
    return values;
}

AreaProfile read_profile_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open profile csv: " + path);
    }

    std::vector<ProfilePoint> points;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        if (line.find("x_m") != std::string::npos) {
            continue;
        }

        std::stringstream ss(line);
        std::string x_text;
        std::string area_text;
        if (!std::getline(ss, x_text, ',')) {
            continue;
        }
        if (!std::getline(ss, area_text, ',')) {
            continue;
        }
        points.push_back(ProfilePoint{std::stod(x_text), std::stod(area_text)});
    }

    return vt_simple::make_profile_from_points(points);
}

void print_help() {
    std::cout
        << "vt_all_solvers_cli\n\n"
        << "Usage:\n"
        << "  --solver cylinder|cone|arma|webster\n"
        << "  --sections N\n"
        << "  --points N\n"
        << "  --f-min-hz F --f-max-hz F\n"
        << "  --grid linear|log\n"
        << "  --beta-loss B        (used by cylinder and arma)\n"
        << "  --rho R --c C\n\n"
        << "Webster FDTD options:\n"
        << "  --signal-sample-rate-hz F\n"
        << "  --signal-duration-s T\n"
        << "  --signal-f0-hz F0\n"
        << "  --signal-f1-hz F1\n"
        << "  --signal-amplitude A\n"
        << "  --spatial-nodes N\n"
        << "  --cfl C\n"
        << "  --observation-node I\n\n"
        << "Geometry input (choose one):\n"
        << "  --profile-csv path.csv\n"
        << "  --geometry three-point --length-m L --areas-m2 a0,a1,a2\n"
        << "  --geometry linear --length-m L --areas-m2 a0,a1\n"
        << "  --geometry uniform-areas --length-m L --areas-m2 a0,a1,...,aN\n"
        << "  --geometry explicit --x-m x0,x1,... --areas-m2 a0,a1,...\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string solver_kind = "cylinder";
        std::string geometry_kind;
        std::string profile_csv_path;
        std::vector<double> x_values_m;
        std::vector<double> area_values_m2;
        double length_m = 0.0;

        std::size_t sections = 20;
        std::size_t points = 256;
        double f_min_hz = 50.0;
        double f_max_hz = 5000.0;
        std::string grid_kind = "linear";
        double beta_loss = 0.0;
        double signal_sample_rate_hz = 48000.0;
        double signal_duration_s = 0.12;
        double signal_f0_hz = -1.0;
        double signal_f1_hz = -1.0;
        double signal_amplitude = 1.0;
        std::size_t spatial_nodes = 21;
        double cfl = 0.95;
        std::size_t observation_node = 0;
        vt_simple::AcousticConstants constants;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_help();
                return 0;
            } else if (arg == "--solver") {
                solver_kind = require_value(i, argc, argv, arg);
            } else if (arg == "--profile-csv") {
                profile_csv_path = require_value(i, argc, argv, arg);
            } else if (arg == "--geometry") {
                geometry_kind = require_value(i, argc, argv, arg);
            } else if (arg == "--x-m") {
                x_values_m = parse_csv_doubles(require_value(i, argc, argv, arg));
            } else if (arg == "--areas-m2") {
                area_values_m2 = parse_csv_doubles(require_value(i, argc, argv, arg));
            } else if (arg == "--length-m") {
                length_m = std::stod(require_value(i, argc, argv, arg));
            } else if (arg == "--sections") {
                sections = static_cast<std::size_t>(std::stoull(require_value(i, argc, argv, arg)));
            } else if (arg == "--points") {
                points = static_cast<std::size_t>(std::stoull(require_value(i, argc, argv, arg)));
            } else if (arg == "--f-min-hz") {
                f_min_hz = std::stod(require_value(i, argc, argv, arg));
            } else if (arg == "--f-max-hz") {
                f_max_hz = std::stod(require_value(i, argc, argv, arg));
            } else if (arg == "--grid") {
                grid_kind = require_value(i, argc, argv, arg);
            } else if (arg == "--beta-loss") {
                beta_loss = std::stod(require_value(i, argc, argv, arg));
            } else if (arg == "--signal-sample-rate-hz") {
                signal_sample_rate_hz = std::stod(require_value(i, argc, argv, arg));
            } else if (arg == "--signal-duration-s") {
                signal_duration_s = std::stod(require_value(i, argc, argv, arg));
            } else if (arg == "--signal-f0-hz") {
                signal_f0_hz = std::stod(require_value(i, argc, argv, arg));
            } else if (arg == "--signal-f1-hz") {
                signal_f1_hz = std::stod(require_value(i, argc, argv, arg));
            } else if (arg == "--signal-amplitude") {
                signal_amplitude = std::stod(require_value(i, argc, argv, arg));
            } else if (arg == "--spatial-nodes") {
                spatial_nodes = static_cast<std::size_t>(std::stoull(require_value(i, argc, argv, arg)));
            } else if (arg == "--cfl") {
                cfl = std::stod(require_value(i, argc, argv, arg));
            } else if (arg == "--observation-node") {
                observation_node = static_cast<std::size_t>(std::stoull(require_value(i, argc, argv, arg)));
            } else if (arg == "--rho") {
                constants.rho_kg_m3 = std::stod(require_value(i, argc, argv, arg));
            } else if (arg == "--c") {
                constants.c_m_s = std::stod(require_value(i, argc, argv, arg));
            } else {
                throw std::invalid_argument("Unknown argument: " + arg);
            }
        }

        AreaProfile profile;
        if (!profile_csv_path.empty()) {
            profile = read_profile_csv(profile_csv_path);
        } else if (geometry_kind == "three-point") {
            if (area_values_m2.size() != 3) {
                throw std::invalid_argument("three-point geometry requires exactly 3 areas.");
            }
            profile = vt_simple::make_three_point_profile(
                length_m,
                area_values_m2[0],
                area_values_m2[1],
                area_values_m2[2]);
        } else if (geometry_kind == "linear") {
            if (area_values_m2.size() != 2) {
                throw std::invalid_argument("linear geometry requires exactly 2 areas.");
            }
            profile = vt_simple::make_linear_profile(length_m, area_values_m2[0], area_values_m2[1]);
        } else if (geometry_kind == "uniform-areas") {
            profile = vt_simple::make_profile_from_areas_uniform(length_m, area_values_m2);
        } else if (geometry_kind == "explicit") {
            profile = vt_simple::make_profile_from_xy(x_values_m, area_values_m2);
        } else {
            throw std::invalid_argument("Please provide either --profile-csv or a valid --geometry.");
        }

        std::vector<double> freqs;
        if (grid_kind == "linear") {
            freqs = vt_simple::make_uniform_frequency_grid(f_min_hz, f_max_hz, points);
        } else if (grid_kind == "log") {
            freqs = vt_simple::make_log_frequency_grid(f_min_hz, f_max_hz, points);
        } else {
            throw std::invalid_argument("--grid must be linear or log.");
        }

        std::vector<vt_simple::FrequencySample> spectrum;
        if (solver_kind == "cylinder") {
            vt_simple::CylinderTlmOptions options;
            options.constants = constants;
            options.beta_loss_np_per_m = beta_loss;
            spectrum = vt_simple::solve_transfer_function_cylinder_tlm(
                profile,
                sections,
                freqs,
                options);
        } else if (solver_kind == "cone") {
            if (f_min_hz <= 0.0) {
                throw std::invalid_argument("cone solver requires positive frequencies: f_min_hz > 0.");
            }
            spectrum = vt_simple::solve_transfer_function_cone_reference(
                profile,
                sections,
                freqs,
                constants);
        } else if (solver_kind == "arma") {
            if (f_min_hz <= 0.0) {
                throw std::invalid_argument("arma solver requires positive frequencies: f_min_hz > 0.");
            }
            spectrum = vt_simple::solve_transfer_function_arma(
                profile,
                sections,
                freqs,
                constants,
                beta_loss);
        } else if (solver_kind == "webster") {
            const double actual_signal_f0_hz =
                (signal_f0_hz >= 0.0) ? signal_f0_hz : f_min_hz;
            const double actual_signal_f1_hz =
                (signal_f1_hz >= 0.0) ? signal_f1_hz : f_max_hz;

            const vt_simple::TimeSignal input_signal =
                vt_simple::make_linear_chirp_signal(
                    signal_sample_rate_hz,
                    signal_duration_s,
                    actual_signal_f0_hz,
                    actual_signal_f1_hz,
                    signal_amplitude);

            vt_simple::WebsterFDTDOptions options;
            options.constants = constants;
            options.spatial_node_count = spatial_nodes;
            options.cfl = cfl;
            options.observation_node = observation_node;

            const vt_simple::WebsterTimeResponse response =
                vt_simple::solve_webster_fdtd(profile, input_signal, options);

            spectrum = vt_simple::estimate_transfer_function_from_signals(
                response.input_pressure,
                response.output_pressure,
                response.sample_rate_hz,
                freqs);
        } else {
            throw std::invalid_argument("--solver must be cylinder, cone, arma, or webster.");
        }

        std::cout << std::fixed << std::setprecision(9);
        std::cout << "solver,frequency_hz,real,imag,magnitude,phase_rad\n";
        for (const auto& sample : spectrum) {
            std::cout << solver_kind << ','
                      << sample.frequency_hz << ','
                      << sample.transfer.real() << ','
                      << sample.transfer.imag() << ','
                      << std::abs(sample.transfer) << ','
                      << std::arg(sample.transfer) << '\n';
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
