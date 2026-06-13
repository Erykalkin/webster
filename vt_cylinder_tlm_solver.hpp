#ifndef VT_CYLINDER_TLM_SOLVER_HPP
#define VT_CYLINDER_TLM_SOLVER_HPP

#include <vector>

#include "vt_geometry_simple.hpp"
#include "vt_solver_common.hpp"

namespace vt_simple {

struct CylinderTlmOptions {
    AcousticConstants constants;
    double beta_loss_np_per_m = 0.0;
};

struct CylinderTransferField {
    std::vector<double> x_grid_m;
    std::vector<double> frequencies_hz;
    std::vector<std::vector<std::complex<double>>> transfer_by_position;
};

// Строит равномерную сетку частот в Гц.
// f_min_hz - нижняя граница, Гц.
// f_max_hz - верхняя граница, Гц.
// point_count - число точек в сетке.
std::vector<double> make_uniform_frequency_grid(double f_min_hz,
                                                double f_max_hz,
                                                std::size_t point_count);

// Строит логарифмическую сетку частот в Гц.
// f_min_hz - нижняя граница, Гц. Должна быть > 0.
// f_max_hz - верхняя граница, Гц.
// point_count - число точек в сетке.
std::vector<double> make_log_frequency_grid(double f_min_hz,
                                            double f_max_hz,
                                            std::size_t point_count);

// Считает передаточную функцию H(f) для профиля трубы.
// Внутри профиль аппроксимируется равномерными цилиндрическими секциями.
// Выходная нагрузка принудительно обнуляется: Zrad = 0.
// Поэтому формула упрощается до H = 1 / A.
//
// profile - исходный профиль трубы.
// section_count - число цилиндрических секций для аппроксимации.
// frequencies_hz - сетка частот, Гц.
// constants - физические параметры среды.
// beta_loss_np_per_m - коэффициент потерь beta, 1/м.
std::vector<FrequencySample> solve_transfer_function_cylinder_tlm(
    const AreaProfile& profile,
    std::size_t section_count,
    const std::vector<double>& frequencies_hz,
    const AcousticConstants& constants,
    double beta_loss_np_per_m = 0.0);

// То же, что и функция выше, но параметры solver'а передаются через options.
std::vector<FrequencySample> solve_transfer_function_cylinder_tlm(
    const AreaProfile& profile,
    std::size_t section_count,
    const std::vector<double>& frequencies_hz,
    const CylinderTlmOptions& options);

// Строит H_k(f) для каждой координаты x_k равномерного цилиндрического разбиения.
// На узле x_k берется та же модель, что и в обычном solver:
// подканал [0, x_k] рассматривается как отдельная труба с нулевой нагрузкой на выходе.
// Поэтому для последнего узла результат совпадает с solve_transfer_function_cylinder_tlm(...).
CylinderTransferField solve_transfer_function_cylinder_tlm_field(
    const AreaProfile& profile,
    std::size_t section_count,
    const std::vector<double>& frequencies_hz,
    const AcousticConstants& constants,
    double beta_loss_np_per_m = 0.0);

CylinderTransferField solve_transfer_function_cylinder_tlm_field(
    const AreaProfile& profile,
    std::size_t section_count,
    const std::vector<double>& frequencies_hz,
    const CylinderTlmOptions& options);

}  // namespace vt_simple

#endif  // VT_CYLINDER_TLM_SOLVER_HPP
