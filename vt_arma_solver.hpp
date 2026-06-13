#ifndef VT_ARMA_SOLVER_HPP
#define VT_ARMA_SOLVER_HPP

#include <complex>
#include <vector>

#include "vt_geometry_simple.hpp"

namespace vt_simple {

// Один отсчет спектра передаточной функции.
struct FrequencySample;

// Полином по z^{-1}.
// coeffs[k] хранит коэффициент при z^{-k}.
using Polynomial = std::vector<double>;

// ARMA-модель для неразветвленной цилиндрической трубы.
struct ArmaTubeModel {
    // sections - равнодлинные цилиндрические секции исходной трубы.
    // Нужны для частотно-зависимого пересчета потерь при вычислении H(f).
    std::vector<CylinderSection> sections;

    // A_poly - полином A(z) в матричной формуле.
    Polynomial A_poly;

    // B_poly - полином B(z) в матричной формуле.
    Polynomial B_poly;

    // C_poly - полином C(z) в матричной формуле.
    Polynomial C_poly;

    // D_poly - полином D(z) в матричной формуле.
    Polynomial D_poly;

    // section_length_m - длина одной цилиндрической секции, м.
    double section_length_m = 0.0;

    // constants - физические параметры среды.
    AcousticConstants constants;
};

// Строит ARMA-модель из цилиндрических секций одинаковой длины.
// profile - исходный профиль трубы.
// section_count - число цилиндрических секций одинаковой длины.
// constants - физические параметры среды.
// beta_loss_np_per_m - одинаковый коэффициент потерь для всех секций.
ArmaTubeModel build_arma_model_from_profile(
    const AreaProfile& profile,
    std::size_t section_count,
    const AcousticConstants& constants,
    double beta_loss_np_per_m = 0.0);

// Вычисляет передаточную функцию ARMA-модели на одной частоте.
// model - ранее построенная ARMA-модель.
// frequency_hz - частота, Гц.
FrequencySample evaluate_arma_at_frequency(const ArmaTubeModel& model,
                                           double frequency_hz);

// Вычисляет передаточную функцию ARMA-модели на всей сетке частот.
// model - ранее построенная ARMA-модель.
// frequencies_hz - сетка частот, Гц.
std::vector<FrequencySample> evaluate_arma_transfer_function(
    const ArmaTubeModel& model,
    const std::vector<double>& frequencies_hz);

// Считает передаточную функцию H(f) для профиля трубы.
// Внутри профиль аппроксимируется равномерными цилиндрическими секциями.
// Выходная нагрузка принудительно обнуляется: Zrad = 0.
// Поэтому формула упрощается до H = 1 / A(z).
//
// profile - исходный профиль трубы.
// section_count - число цилиндрических секций одинаковой длины.
// frequencies_hz - сетка частот, Гц.
// constants - физические параметры среды.
// beta_loss_np_per_m - коэффициент потерь beta, 1/м.
std::vector<FrequencySample> solve_transfer_function_arma(
    const AreaProfile& profile,
    std::size_t section_count,
    const std::vector<double>& frequencies_hz,
    const AcousticConstants& constants,
    double beta_loss_np_per_m = 0.0);

}  // namespace vt_simple

#endif  // VT_ARMA_SOLVER_HPP
