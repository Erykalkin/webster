#ifndef VT_CONE_REFERENCE_SOLVER_HPP
#define VT_CONE_REFERENCE_SOLVER_HPP

#include <vector>

#include "vt_geometry_simple.hpp"
#include "vt_solver_common.hpp"

namespace vt_simple {

// Считает передаточную функцию H(f) для профиля трубы.
// Внутри профиль аппроксимируется равномерными коническими секциями.
// Выходная нагрузка принудительно обнуляется: Zrad = 0.
// Поэтому формула упрощается до H = 1 / A.
//
// profile - исходный профиль трубы.
// section_count - число конических секций для аппроксимации.
// frequencies_hz - сетка частот, Гц.
// constants - физические параметры среды.
std::vector<FrequencySample> solve_transfer_function_cone_reference(
    const AreaProfile& profile,
    std::size_t section_count,
    const std::vector<double>& frequencies_hz,
    const AcousticConstants& constants);

}  // namespace vt_simple

#endif  // VT_CONE_REFERENCE_SOLVER_HPP
