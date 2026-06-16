#ifndef VT_SOLVER_COMMON_HPP
#define VT_SOLVER_COMMON_HPP

#include <complex>

namespace vt_simple {

// Один отсчет спектра передаточной функции.
struct FrequencySample {
    // frequency_hz - частота, Гц.
    double frequency_hz = 0.0;

    // transfer - значение передаточной функции H(f) на этой частоте.
    std::complex<double> transfer = std::complex<double>(0.0, 0.0);
};

}  // namespace vt_simple

#endif  // VT_SOLVER_COMMON_HPP
