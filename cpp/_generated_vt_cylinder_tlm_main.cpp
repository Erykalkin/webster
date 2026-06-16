#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "vt_cylinder_tlm_solver.hpp"
#include "vt_geometry_simple.hpp"

int main() {
    vt_simple::AcousticConstants acoustics;
    acoustics.rho_kg_m3 = 1.225;
    acoustics.c_m_s = 343.0;

    const vt_simple::AreaProfile profile = vt_simple::make_three_point_profile(
        0.17,
        3.0e-4,
        8.0e-5,
        4.0e-4
    );

    const std::vector<double> freqs = vt_simple::make_uniform_frequency_grid(
        50.0,
        5000.0,
        4
    );

    const auto spectrum = vt_simple::solve_transfer_function_cylinder_tlm(
        profile,
        2,
        freqs,
        acoustics,
        0.0
    );

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "f_Hz  |  |H(f)|\n";
    std::cout << "----------------\n";

    for (const auto& sample : spectrum) {
        std::cout << std::setw(6) << sample.frequency_hz << "  |  "
                  << std::setw(8) << std::abs(sample.transfer) << "\n";
    }

    return 0;
}
