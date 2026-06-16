#include <iomanip>
#include <iostream>

#include "vt_arma_solver.hpp"
#include "vt_cone_reference_solver.hpp"
#include "vt_cylinder_tlm_solver.hpp"
#include "vt_geometry_simple.hpp"

int main() {
    try {
        using namespace vt_simple;

        // Создаем простой профиль трубы для демонстрации.
        const AreaProfile profile = make_three_point_profile(
            0.17,
            1.2e-4,
            3.0e-4,
            2.0e-4);

        // Строим равномерную сетку частот.
        const std::vector<double> frequencies_hz = make_uniform_frequency_grid(
            100.0,
            3000.0,
            8);

        // Физические параметры воздуха.
        AcousticConstants constants;
        constants.rho_kg_m3 = 1.225;
        constants.c_m_s = 343.0;

        // Считаем передаточную функцию тремя способами.
        const auto cylinder_spectrum = solve_transfer_function_cylinder_tlm(
            profile,
            20,
            frequencies_hz,
            constants,
            0.0);

        const auto cone_spectrum = solve_transfer_function_cone_reference(
            profile,
            20,
            frequencies_hz,
            constants);

        const auto arma_spectrum = solve_transfer_function_arma(
            profile,
            20,
            frequencies_hz,
            constants,
            0.0);

        // Выводим несколько первых значений для проверки.
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "f_hz\t|H_cyl|\t|H_cone|\t|H_arma|\n";

        for (std::size_t i = 0; i < frequencies_hz.size(); ++i) {
            std::cout << frequencies_hz[i] << '\t'
                      << std::abs(cylinder_spectrum[i].transfer) << '\t'
                      << std::abs(cone_spectrum[i].transfer) << '\t'
                      << std::abs(arma_spectrum[i].transfer) << '\n';
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}
