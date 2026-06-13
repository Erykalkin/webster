#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "vt_cylinder_tlm_solver.hpp"
#include "vt_geometry_simple.hpp"
#include "vt_webster_fdtd_solver.hpp"

int main() {
    try {
        using namespace vt_simple;

        const AreaProfile profile = make_three_point_profile(
            0.17,
            3.0e-4,
            8.0e-5,
            4.0e-4);

        AcousticConstants acoustics;
        acoustics.rho_kg_m3 = 1.225;
        acoustics.c_m_s = 343.0;

        const std::vector<double> frequencies_hz =
            make_uniform_frequency_grid(50.0, 5000.0, 256);

        const TimeSignal input_signal = make_linear_chirp_signal(
            48000.0,
            0.12,
            50.0,
            5000.0,
            1.0);

        WebsterFDTDOptions webster_options;
        webster_options.constants = acoustics;
        webster_options.spatial_node_count = 21;
        webster_options.cfl = 0.98;
        webster_options.observation_node = 0;

        const WebsterTimeResponse time_response =
            solve_webster_fdtd(profile, input_signal, webster_options);

        const std::vector<FrequencySample> webster_spectrum =
            estimate_transfer_function_from_signals(
                time_response.input_pressure,
                time_response.output_pressure,
                time_response.sample_rate_hz,
                frequencies_hz);

        const std::vector<FrequencySample> cylinder_spectrum =
            solve_transfer_function_cylinder_tlm(
                profile,
                120,
                frequencies_hz,
                acoustics,
                0.0);

        double max_webster = 0.0;
        double max_cylinder = 0.0;
        for (std::size_t i = 0; i < frequencies_hz.size(); ++i) {
            max_webster = std::max(max_webster,
                                   std::abs(webster_spectrum[i].transfer));
            max_cylinder = std::max(max_cylinder,
                                    std::abs(cylinder_spectrum[i].transfer));
        }

        std::ofstream csv("compare_webster_vs_cylinder.csv");
        csv << "frequency_hz,webster_mag,webster_mag_norm,"
               "cylinder_mag,cylinder_mag_norm\n";

        for (std::size_t i = 0; i < frequencies_hz.size(); ++i) {
            const double webster_mag =
                std::abs(webster_spectrum[i].transfer);
            const double cylinder_mag =
                std::abs(cylinder_spectrum[i].transfer);

            const double webster_mag_norm =
                (max_webster > 0.0) ? (webster_mag / max_webster) : 0.0;
            const double cylinder_mag_norm =
                (max_cylinder > 0.0) ? (cylinder_mag / max_cylinder) : 0.0;

            csv << std::fixed << std::setprecision(9)
                << frequencies_hz[i] << ','
                << webster_mag << ','
                << webster_mag_norm << ','
                << cylinder_mag << ','
                << cylinder_mag_norm << '\n';
        }

        std::ofstream gp("plot_compare_webster_vs_cylinder.gp");
        gp << "set datafile separator ','\n";
        gp << "set key left top\n";
        gp << "set grid\n";
        gp << "set xlabel 'f, Hz'\n";
        gp << "set ylabel 'Normalized magnitude'\n";
        gp << "set title 'Classical Webster FDTD vs Cylinder TLM'\n";
        gp << "plot "
           << "'compare_webster_vs_cylinder.csv' using 1:3 with lines lw 2 "
              "title 'Webster FDTD', "
           << "'compare_webster_vs_cylinder.csv' using 1:5 with lines lw 2 "
              "title 'Cylinder TLM'\n";

        std::cout << "Saved compare_webster_vs_cylinder.csv\n";
        std::cout << "Saved plot_compare_webster_vs_cylinder.gp\n";
        std::cout << "If gnuplot is installed, run:\n";
        std::cout << "  gnuplot -persist plot_compare_webster_vs_cylinder.gp\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
