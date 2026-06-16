#include "vt_cylinder_tlm_solver.hpp"

#include <cmath>
#include <stdexcept>

namespace vt_simple {
namespace {

using Complex = std::complex<double>;

struct LossySectionState {
    Complex lambda = Complex(0.0, 0.0);
    Complex dzeta = Complex(1.0, 0.0);
};

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

// Строит матрицу одной цилиндрической секции.
// Формула соответствует стандартной матрице длинной линии.
// omega_rad_s - круговая частота, рад/с.
// sec - параметры секции.
// constants - физические параметры воздуха.
LossySectionState make_lossy_section_state(double omega_rad_s,
                                           const CylinderSection& sec,
                                           const AcousticConstants& constants) {
    if (!(sec.length_m > 0.0)) {
        throw std::invalid_argument("CylinderSection.length_m must be positive.");
    }
    if (!(sec.area_m2 > 0.0)) {
        throw std::invalid_argument("CylinderSection.area_m2 must be positive.");
    }

    const Complex j(0.0, 1.0);
    LossySectionState out;
    out.lambda = j * (omega_rad_s / constants.c_m_s);
    out.dzeta = Complex(1.0, 0.0);

    if (sec.beta_loss_np_per_m > 0.0) {
        const double viscous_scale = sec.beta_loss_np_per_m * constants.c_m_s;
        const Complex alpha = std::sqrt(j * omega_rad_s * viscous_scale);

        // В текущей упрощенной модели считаем стенки жесткими, поэтому hi = alpha.
        out.lambda = (alpha + j * omega_rad_s) / constants.c_m_s;
        out.dzeta = Complex(1.0, 0.0);
    }

    return out;
}

Mat2 cylinder_section_matrix(double omega_rad_s,
                             const CylinderSection& sec,
                             const AcousticConstants& constants) {
    // Волновое сопротивление воздуха rho*c.
    const double zc = constants.rho_kg_m3 * constants.c_m_s;
    const LossySectionState loss =
        make_lossy_section_state(omega_rad_s, sec, constants);
    const Complex gl = loss.lambda * sec.length_m;

    // Гиперболические функции описывают распространение волны по секции.
    const Complex ch = std::cosh(gl);
    const Complex sh = std::sinh(gl);
    const Complex zc_eff = (zc * loss.dzeta) / sec.area_m2;
    const Complex yc_eff = Complex(1.0, 0.0) / zc_eff;

    Mat2 out;

    // Диагональные элементы отвечают за "прямой проход" через секцию.
    out.a11 = ch;
    out.a22 = ch;

    // Внедиагональные элементы связывают давление и объемную скорость.
    out.a12 = -zc_eff * sh;
    out.a21 = -yc_eff * sh;

    return out;
}

}  // namespace

std::vector<double> make_uniform_frequency_grid(double f_min_hz,
                                                double f_max_hz,
                                                std::size_t point_count) {
    // В сетке должно быть хотя бы две точки.
    if (point_count < 2) {
        throw std::invalid_argument("point_count must be >= 2.");
    }

    // Верхняя граница должна быть больше нижней.
    if (!(f_max_hz > f_min_hz)) {
        throw std::invalid_argument("f_max_hz must be greater than f_min_hz.");
    }

    std::vector<double> frequencies(point_count, 0.0);

    // Равномерный шаг сетки по частоте.
    const double df = (f_max_hz - f_min_hz) /
                      static_cast<double>(point_count - 1);

    for (std::size_t i = 0; i < point_count; ++i) {
        // Заполняем очередную точку сетки.
        frequencies[i] = f_min_hz + static_cast<double>(i) * df;
    }

    return frequencies;
}

std::vector<double> make_log_frequency_grid(double f_min_hz,
                                            double f_max_hz,
                                            std::size_t point_count) {
    if (point_count < 2) {
        throw std::invalid_argument("point_count must be >= 2.");
    }
    if (!(f_min_hz > 0.0)) {
        throw std::invalid_argument("f_min_hz must be positive for a log grid.");
    }
    if (!(f_max_hz > f_min_hz)) {
        throw std::invalid_argument("f_max_hz must be greater than f_min_hz.");
    }

    std::vector<double> frequencies(point_count, 0.0);
    const double log_min = std::log(f_min_hz);
    const double log_max = std::log(f_max_hz);
    const double dlog =
        (log_max - log_min) / static_cast<double>(point_count - 1);

    for (std::size_t i = 0; i < point_count; ++i) {
        frequencies[i] = std::exp(log_min + static_cast<double>(i) * dlog);
    }

    return frequencies;
}

std::vector<FrequencySample> solve_transfer_function_cylinder_tlm(
    const AreaProfile& profile,
    std::size_t section_count,
    const std::vector<double>& frequencies_hz,
    const AcousticConstants& constants,
    double beta_loss_np_per_m) {

    // Проверяем профиль на корректность.
    validate_profile(profile);

    // Нельзя решать задачу без частотной сетки.
    if (frequencies_hz.empty()) {
        throw std::invalid_argument("frequencies_hz must not be empty.");
    }

    // Сначала аппроксимируем исходный профиль цилиндрами.
    const std::vector<CylinderSection> sections = build_cylinders_uniform(
        profile,
        section_count,
        beta_loss_np_per_m);

    std::vector<FrequencySample> spectrum;
    spectrum.reserve(frequencies_hz.size());

    // Идем по всем частотам и считаем H(f) отдельно на каждой частоте.
    for (double frequency_hz : frequencies_hz) {
        // Частота должна быть неотрицательной.
        if (frequency_hz < 0.0) {
            throw std::invalid_argument("frequency_hz must be non-negative.");
        }

        // Переходим от частоты в Гц к круговой частоте в рад/с.
        const double omega_rad_s = 2.0 * kPi * frequency_hz;

        // Начинаем с единичной матрицы всей трубы.
        Mat2 global;

        // Последовательно домножаем матрицы всех цилиндрических секций.
        for (const CylinderSection& sec : sections) {
            const Mat2 local = cylinder_section_matrix(omega_rad_s, sec, constants);
            global = multiply(local, global);
        }

        // Выделяем элементы итоговой матрицы всей трубы.
        const Complex A = global.a11;

        // Здесь мы делаем ровно то, о чем ты просил:
        // принудительно обнуляем нагрузку на выходе трубы.
        // То есть Zrad = 0, и передаточная функция упрощается до H = 1 / A.
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

std::vector<FrequencySample> solve_transfer_function_cylinder_tlm(
    const AreaProfile& profile,
    std::size_t section_count,
    const std::vector<double>& frequencies_hz,
    const CylinderTlmOptions& options) {
    return solve_transfer_function_cylinder_tlm(
        profile,
        section_count,
        frequencies_hz,
        options.constants,
        options.beta_loss_np_per_m);
}

CylinderTransferField solve_transfer_function_cylinder_tlm_field(
    const AreaProfile& profile,
    std::size_t section_count,
    const std::vector<double>& frequencies_hz,
    const AcousticConstants& constants,
    double beta_loss_np_per_m) {

    validate_profile(profile);

    if (frequencies_hz.empty()) {
        throw std::invalid_argument("frequencies_hz must not be empty.");
    }

    const std::vector<CylinderSection> sections = build_cylinders_uniform(
        profile,
        section_count,
        beta_loss_np_per_m);

    CylinderTransferField field;
    field.frequencies_hz = frequencies_hz;
    field.x_grid_m.resize(section_count + 1, 0.0);
    field.transfer_by_position.assign(
        section_count + 1,
        std::vector<Complex>(frequencies_hz.size(), Complex(0.0, 0.0)));

    const double x0 = profile.points.front().x_m;
    const double total_length_m = profile.points.back().x_m - x0;
    const double dx = total_length_m / static_cast<double>(section_count);

    for (std::size_t i = 0; i <= section_count; ++i) {
        field.x_grid_m[i] = x0 + static_cast<double>(i) * dx;
    }

    for (std::size_t fi = 0; fi < frequencies_hz.size(); ++fi) {
        const double frequency_hz = frequencies_hz[fi];
        if (frequency_hz < 0.0) {
            throw std::invalid_argument("frequency_hz must be non-negative.");
        }

        const double omega_rad_s = 2.0 * kPi * frequency_hz;
        Mat2 global;

        field.transfer_by_position[0][fi] = Complex(1.0, 0.0);

        for (std::size_t si = 0; si < sections.size(); ++si) {
            const Mat2 local =
                cylinder_section_matrix(omega_rad_s, sections[si], constants);
            global = multiply(local, global);
            field.transfer_by_position[si + 1][fi] =
                Complex(1.0, 0.0) / global.a11;
        }
    }

    return field;
}

CylinderTransferField solve_transfer_function_cylinder_tlm_field(
    const AreaProfile& profile,
    std::size_t section_count,
    const std::vector<double>& frequencies_hz,
    const CylinderTlmOptions& options) {
    return solve_transfer_function_cylinder_tlm_field(
        profile,
        section_count,
        frequencies_hz,
        options.constants,
        options.beta_loss_np_per_m);
}

}  // namespace vt_simple
