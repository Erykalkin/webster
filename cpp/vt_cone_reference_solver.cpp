#include "vt_cone_reference_solver.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>

namespace vt_simple {
namespace {

using Complex = std::complex<double>;

// Простая 2x2 матрица комплексных чисел.
struct Mat2 {
    // a11, a12, a21, a22 - элементы матрицы.
    Complex a11 = Complex(1.0, 0.0);
    Complex a12 = Complex(0.0, 0.0);
    Complex a21 = Complex(0.0, 0.0);
    Complex a22 = Complex(1.0, 0.0);
};

// Умножение двух матриц 2x2.
Mat2 multiply(const Mat2& lhs, const Mat2& rhs) {
    Mat2 out;

    // Первая строка результата.
    out.a11 = lhs.a11 * rhs.a11 + lhs.a12 * rhs.a21;
    out.a12 = lhs.a11 * rhs.a12 + lhs.a12 * rhs.a22;

    // Вторая строка результата.
    out.a21 = lhs.a21 * rhs.a11 + lhs.a22 * rhs.a21;
    out.a22 = lhs.a21 * rhs.a12 + lhs.a22 * rhs.a22;

    return out;
}

// Строит матрицу одной конической секции по базовой lossless-модели.
// Формула соответствует произведению Phi * Gamma * H.
// omega_rad_s - круговая частота, рад/с.
// sec - параметры конической секции.
// constants - физические параметры воздуха.
Mat2 cone_section_matrix_lossless(double omega_rad_s,
                                  const ConeSection& sec,
                                  const AcousticConstants& constants) {
    // Проверяем длину секции.
    if (!(sec.length_m > 0.0)) {
        throw std::invalid_argument("ConeSection.length_m must be positive.");
    }
    if (!(sec.area_in_m2 > 0.0)) {
        throw std::invalid_argument("ConeSection.area_in_m2 must be positive.");
    }
    if (!(sec.area_out_m2 > 0.0)) {
        throw std::invalid_argument("ConeSection.area_out_m2 must be positive.");
    }
    if (!(omega_rad_s > 0.0)) {
        throw std::invalid_argument("omega_rad_s must be positive for the cone solver.");
    }

    // Переводим площади во входной и выходной радиусы.
    const double r_in_m = radius_from_area(sec.area_in_m2);
    const double r_out_m = radius_from_area(sec.area_out_m2);

    // Коэффициент раскрытия конуса beta = (R - r) / (r * l).
    const double beta_opening =
        (r_out_m - r_in_m) / (r_in_m * sec.length_m);

    // Волновое сопротивление воздуха rho*c.
    const double zc = constants.rho_kg_m3 * constants.c_m_s;

    // Для lossless модели gamma = j*omega/c.
    const Complex gamma(0.0, omega_rad_s / constants.c_m_s);

    // Аргумент гиперболических функций.
    const Complex gl = gamma * sec.length_m;
    const Complex ch = std::cosh(gl);
    const Complex sh = std::sinh(gl);

    // Матрица Phi из статьи.
    Mat2 phi;
    phi.a11 = 1.0 / (1.0 + beta_opening * sec.length_m);
    phi.a12 = Complex(0.0, 0.0);
    phi.a21 = (beta_opening * sec.area_in_m2) / (gamma * zc);
    phi.a22 = sec.area_in_m2 * (1.0 + beta_opening * sec.length_m);

    // Матрица Gamma из статьи.
    Mat2 gamma_m;
    gamma_m.a11 = ch;
    gamma_m.a12 = -zc * sh;
    gamma_m.a21 = -(1.0 / zc) * sh;
    gamma_m.a22 = ch;

    // Матрица H из статьи.
    Mat2 h;
    h.a11 = Complex(1.0, 0.0);
    h.a12 = Complex(0.0, 0.0);
    h.a21 = -beta_opening / (gamma * zc);
    h.a22 = 1.0 / sec.area_in_m2;

    // Полная матрица одной конической секции.
    return multiply(multiply(phi, gamma_m), h);
}

}  // namespace

std::vector<FrequencySample> solve_transfer_function_cone_reference(
    const AreaProfile& profile,
    std::size_t section_count,
    const std::vector<double>& frequencies_hz,
    const AcousticConstants& constants) {

    // Проверяем профиль на корректность.
    validate_profile(profile);

    // Нельзя решать задачу без частотной сетки.
    if (frequencies_hz.empty()) {
        throw std::invalid_argument("frequencies_hz must not be empty.");
    }

    // Сначала аппроксимируем исходный профиль конусами.
    const std::vector<ConeSection> sections = build_cones_uniform(profile, section_count);

    std::vector<FrequencySample> spectrum;
    spectrum.reserve(frequencies_hz.size());

    // Идем по всем частотам и считаем H(f) отдельно на каждой частоте.
    for (double frequency_hz : frequencies_hz) {
        // Частота должна быть положительной.
        if (!(frequency_hz > 0.0)) {
            throw std::invalid_argument("frequency_hz must be positive for the cone solver.");
        }

        // Переходим от частоты в Гц к круговой частоте в рад/с.
        const double omega_rad_s = 2.0 * kPi * frequency_hz;

        // Начинаем с единичной матрицы всей трубы.
        Mat2 global;

        // Последовательно домножаем матрицы всех конических секций.
        for (const ConeSection& sec : sections) {
            const Mat2 local = cone_section_matrix_lossless(omega_rad_s, sec, constants);
            global = multiply(local, global);
        }

        // Выделяем элемент A итоговой матрицы всей трубы.
        const Complex A = global.a11;

        // Как и в минимальном цилиндрическом solver'е, жестко ставим Zrad = 0.
        // Поэтому передаточная функция упрощается до H = 1 / A.
        const Complex H = Complex(1.0, 0.0) / A;

        FrequencySample sample;

        // Сохраняем частоту текущего отсчета.
        sample.frequency_hz = frequency_hz;

        // Сохраняем рассчитанное значение передаточной функции.
        sample.transfer = H;

        spectrum.push_back(sample);
    }

    return spectrum;
}

}  // namespace vt_simple
