#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "vt_cylinder_tlm_solver.hpp"
#include "vt_geometry_simple.hpp"

int main() {
    // Задаем физические параметры воздуха.
    vt_simple::AcousticConstants acoustics;
    acoustics.rho_kg_m3 = 1.225;
    acoustics.c_m_s = 343.0;

    // Строим простой тестовый профиль трубы длиной 17 см.
    // На входе площадь больше, в середине есть сужение, на выходе снова расширение.
    const vt_simple::AreaProfile profile = vt_simple::make_three_point_profile(
        0.17,       // length_m
        3.0e-4,     // area_left_m2   = 3.0 см^2
        8.0e-5,     // area_middle_m2 = 0.8 см^2
        4.0e-4      // area_right_m2  = 4.0 см^2
    );

    // Строим сетку частот от 50 до 5000 Гц.
    const std::vector<double> freqs = vt_simple::make_uniform_frequency_grid(
        50.0,       // f_min_hz
        5000.0,     // f_max_hz
        32          // point_count
    );

    // Считаем передаточную функцию для цилиндрической аппроксимации.
    // Внутри solver использует Zrad = 0, то есть "губы" обнулены.
    const auto spectrum = vt_simple::solve_transfer_function_cylinder_tlm(
        profile,    // профиль трубы
        20,         // число цилиндрических секций
        freqs,      // частотная сетка
        acoustics,  // физические параметры среды
        0.0         // коэффициент потерь beta_loss_np_per_m
    );

    // Печатаем модуль передаточной функции на каждой частоте.
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "f_Hz  |  |H(f)|\n";
    std::cout << "----------------\n";

    for (const auto& sample : spectrum) {
        std::cout << std::setw(6) << sample.frequency_hz << "  |  "
                  << std::setw(8) << std::abs(sample.transfer) << "\n";
    }

    return 0;
}
