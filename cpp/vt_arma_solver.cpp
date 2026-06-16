#include "vt_arma_solver.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>

#include "vt_cylinder_tlm_solver.hpp"

namespace vt_simple {
namespace {

using Complex = std::complex<double>;

struct LossySectionState {
    Complex lambda = Complex(0.0, 0.0);
    Complex dzeta = Complex(1.0, 0.0);
};

struct Mat2 {
    Complex a11 = Complex(1.0, 0.0);
    Complex a12 = Complex(0.0, 0.0);
    Complex a21 = Complex(0.0, 0.0);
    Complex a22 = Complex(1.0, 0.0);
};

// Удаляет незначимые хвостовые коэффициенты полинома.
// poly - изменяемый полином.
void trim_trailing_zeros(Polynomial& poly) {
    // Пока в хвосте больше одного коэффициента и последний почти нулевой,
    // удаляем его, чтобы не раздувать степень полинома.
    while (poly.size() > 1 && std::abs(poly.back()) < 1e-15) {
        poly.pop_back();
    }
}

// Складывает два полинома по z^{-1}.
// lhs - первый полином.
// rhs - второй полином.
Polynomial poly_add(const Polynomial& lhs, const Polynomial& rhs) {
    // Степень суммы не больше максимальной из двух степеней.
    Polynomial out(std::max(lhs.size(), rhs.size()), 0.0);

    // Прибавляем коэффициенты первого полинома.
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        out[i] += lhs[i];
    }

    // Прибавляем коэффициенты второго полинома.
    for (std::size_t i = 0; i < rhs.size(); ++i) {
        out[i] += rhs[i];
    }

    // Убираем нулевой хвост после сложения.
    trim_trailing_zeros(out);
    return out;
}

// Умножает полином на аффинный множитель c0 + c1 * z^{-1}.
// poly - исходный полином.
// c0 - коэффициент при z^0.
// c1 - коэффициент при z^{-1}.
Polynomial poly_mul_affine(const Polynomial& poly, double c0, double c1) {
    // После умножения на линейный множитель степень увеличится максимум на 1.
    Polynomial out(poly.size() + 1, 0.0);

    for (std::size_t k = 0; k < poly.size(); ++k) {
        // Добавляем вклад c0 * poly.
        out[k] += c0 * poly[k];

        // Добавляем вклад c1 * z^{-1} * poly.
        out[k + 1] += c1 * poly[k];
    }

    // Чистим незначимый хвост.
    trim_trailing_zeros(out);
    return out;
}

// Строит частотно-зависимые параметры секции по мотивам формул из sol.ipynb.
// beta_loss_np_per_m интерпретируем как базовый коэффициент потерь на длину;
// затем переводим его в "вязкий" параметр масштаба c * beta, чтобы сохранить размерность.
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

        // В текущем минимальном solver'е считаем стенки жесткими, поэтому hi = alpha.
        out.lambda = (alpha + j * omega_rad_s) / constants.c_m_s;
        out.dzeta = Complex(1.0, 0.0);
    }

    return out;
}

Mat2 multiply(const Mat2& lhs, const Mat2& rhs) {
    Mat2 out;
    out.a11 = lhs.a11 * rhs.a11 + lhs.a12 * rhs.a21;
    out.a12 = lhs.a11 * rhs.a12 + lhs.a12 * rhs.a22;
    out.a21 = lhs.a21 * rhs.a11 + lhs.a22 * rhs.a21;
    out.a22 = lhs.a21 * rhs.a12 + lhs.a22 * rhs.a22;
    return out;
}

Mat2 cylinder_section_matrix(double omega_rad_s,
                             const CylinderSection& sec,
                             const AcousticConstants& constants) {
    const double zc = constants.rho_kg_m3 * constants.c_m_s;
    const LossySectionState loss =
        make_lossy_section_state(omega_rad_s, sec, constants);
    const Complex gl = loss.lambda * sec.length_m;
    const Complex ch = std::cosh(gl);
    const Complex sh = std::sinh(gl);
    const Complex zc_eff = (zc * loss.dzeta) / sec.area_m2;
    const Complex yc_eff = Complex(1.0, 0.0) / zc_eff;

    Mat2 out;
    out.a11 = ch;
    out.a22 = ch;
    out.a12 = -zc_eff * sh;
    out.a21 = -yc_eff * sh;
    return out;
}

// Строит ARMA-модель из уже готовых цилиндрических секций.
// sections - цилиндрические секции одинаковой длины.
// constants - физические параметры среды.
ArmaTubeModel build_arma_model_from_cylinders(const std::vector<CylinderSection>& sections,
                                              const AcousticConstants& constants) {
    // Пустой набор секций недопустим.
    if (sections.empty()) {
        throw std::invalid_argument("sections must not be empty.");
    }

    // ARMA-вывод требует одинаковой длины всех цилиндров.
    const double section_length_m = sections.front().length_m;
    if (!(section_length_m > 0.0)) {
        throw std::invalid_argument("section_length_m must be positive.");
    }

    for (const CylinderSection& sec : sections) {
        // Проверяем одинаковость длин секций.
        if (std::abs(sec.length_m - section_length_m) > 1e-12) {
            throw std::invalid_argument(
                "All cylinder sections must have the same length for the ARMA solver.");
        }

        // Проверяем физическую допустимость площади.
        if (!(sec.area_m2 > 0.0)) {
            throw std::invalid_argument("CylinderSection.area_m2 must be positive.");
        }
    }

    // Начинаем с единичной матрицы M_0 = I в полиномиальной форме.
    Polynomial A{1.0};
    Polynomial B{0.0};
    Polynomial C{0.0};
    Polynomial D{1.0};

    // Часто встречающийся множитель rho*c.
    const double zc = constants.rho_kg_m3 * constants.c_m_s;

    for (const CylinderSection& sec : sections) {
        // alpha_i кодирует потери на текущей секции.
        const double alpha = std::exp(-2.0 * sec.beta_loss_np_per_m * sec.length_m);

        // Коэффициенты связи давления и потока для текущей площади.
        const double zc_over_s = zc / sec.area_m2;
        const double s_over_zc = sec.area_m2 / zc;

        // Сохраняем предыдущие полиномы, чтобы новые считались из одного шага.
        const Polynomial A_prev = A;
        const Polynomial B_prev = B;
        const Polynomial C_prev = C;
        const Polynomial D_prev = D;

        // Обновляем полином A_i.
        A = poly_add(poly_mul_affine(A_prev, 1.0, alpha),
                     poly_mul_affine(B_prev, -zc_over_s, zc_over_s * alpha));

        // Обновляем полином B_i.
        B = poly_add(poly_mul_affine(A_prev, -s_over_zc, s_over_zc * alpha),
                     poly_mul_affine(B_prev, 1.0, alpha));

        // Обновляем полином C_i.
        C = poly_add(poly_mul_affine(C_prev, 1.0, alpha),
                     poly_mul_affine(D_prev, -zc_over_s, zc_over_s * alpha));

        // Обновляем полином D_i.
        D = poly_add(poly_mul_affine(C_prev, -s_over_zc, s_over_zc * alpha),
                     poly_mul_affine(D_prev, 1.0, alpha));
    }

    ArmaTubeModel model;
    model.sections = sections;

    // Сохраняем полиномы итоговой модели.
    model.A_poly = A;
    model.B_poly = B;
    model.C_poly = C;
    model.D_poly = D;

    // Запоминаем длину одной секции и физические константы.
    model.section_length_m = section_length_m;
    model.constants = constants;

    return model;
}

}  // namespace

ArmaTubeModel build_arma_model_from_profile(const AreaProfile& profile,
                                            std::size_t section_count,
                                            const AcousticConstants& constants,
                                            double beta_loss_np_per_m) {
    // Проверяем профиль на корректность.
    validate_profile(profile);

    // Сначала аппроксимируем исходный профиль цилиндрами одинаковой длины.
    const std::vector<CylinderSection> sections = build_cylinders_uniform(
        profile,
        section_count,
        beta_loss_np_per_m);

    // Затем строим ARMA-модель из набора цилиндров.
    return build_arma_model_from_cylinders(sections, constants);
}

FrequencySample evaluate_arma_at_frequency(const ArmaTubeModel& model,
                                           double frequency_hz) {
    // Частота должна быть положительной.
    if (!(frequency_hz > 0.0)) {
        throw std::invalid_argument("frequency_hz must be positive.");
    }

    // Длина секции должна быть положительной.
    if (!(model.section_length_m > 0.0)) {
        throw std::invalid_argument("model.section_length_m must be positive.");
    }

    if (model.sections.empty()) {
        throw std::invalid_argument("model.sections must not be empty.");
    }

    // Переводим частоту из Гц в круговую частоту.
    const double omega_rad_s = 2.0 * kPi * frequency_hz;

    Mat2 global;

    for (const CylinderSection& sec : model.sections) {
        const Mat2 local =
            cylinder_section_matrix(omega_rad_s, sec, model.constants);
        global = multiply(local, global);
    }

    const Complex A = global.a11;

    // Как и в минимальном цилиндрическом solver'е, жестко ставим Zrad = 0.
    // Поэтому передаточная функция упрощается до H = 1 / A.
    const Complex H = Complex(1.0, 0.0) / A;

    FrequencySample sample;

    // Сохраняем частоту текущего отсчета.
    sample.frequency_hz = frequency_hz;

    // Сохраняем рассчитанное значение передаточной функции.
    sample.transfer = H;

    return sample;
}

std::vector<FrequencySample> evaluate_arma_transfer_function(
    const ArmaTubeModel& model,
    const std::vector<double>& frequencies_hz) {

    // Нельзя решать задачу без частотной сетки.
    if (frequencies_hz.empty()) {
        throw std::invalid_argument("frequencies_hz must not be empty.");
    }

    std::vector<FrequencySample> spectrum;
    spectrum.reserve(frequencies_hz.size());

    // Идем по всем частотам и считаем H(f) отдельно на каждой частоте.
    for (double frequency_hz : frequencies_hz) {
        spectrum.push_back(evaluate_arma_at_frequency(model, frequency_hz));
    }

    return spectrum;
}

std::vector<FrequencySample> solve_transfer_function_arma(
    const AreaProfile& profile,
    std::size_t section_count,
    const std::vector<double>& frequencies_hz,
    const AcousticConstants& constants,
    double beta_loss_np_per_m) {

    // Сначала строим ARMA-модель из исходного профиля.
    const ArmaTubeModel model = build_arma_model_from_profile(
        profile,
        section_count,
        constants,
        beta_loss_np_per_m);

    // Затем вычисляем передаточную функцию модели на нужной сетке частот.
    return evaluate_arma_transfer_function(model, frequencies_hz);
}

}  // namespace vt_simple
