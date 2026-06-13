#ifndef VT_GEOMETRY_SIMPLE_HPP
#define VT_GEOMETRY_SIMPLE_HPP

#include <cstddef>
#include <vector>

namespace vt_simple {

// Математическая константа pi.
constexpr double kPi = 3.141592653589793238462643383279502884;

// Физические константы среды.
// Все величины задаются в SI.
struct AcousticConstants {
    // rho_kg_m3 - плотность воздуха, кг/м^3.
    double rho_kg_m3 = 1.225;

    // c_m_s - скорость звука, м/с.
    double c_m_s = 343.0;
};

// Одна точка профиля трубы.
struct ProfilePoint {
    // x_m - координата вдоль трубы, м.
    double x_m = 0.0;

    // area_m2 - площадь сечения в этой точке, м^2.
    double area_m2 = 0.0;
};

// Профиль трубы как набор точек.
// Между точками площадь интерполируется линейно.
struct AreaProfile {
    // points - узлы профиля, отсортированные по x.
    std::vector<ProfilePoint> points;
};

// Одна цилиндрическая секция.
// Именно на таких секциях работает наш минимальный solver.
struct CylinderSection {
    // length_m - длина секции, м.
    double length_m = 0.0;

    // area_m2 - постоянная площадь этой секции, м^2.
    double area_m2 = 0.0;

    // beta_loss_np_per_m - коэффициент потерь, 1/м.
    // Если потери не нужны, передаем 0.
    double beta_loss_np_per_m = 0.0;
};

struct ConeSection {
    double length_m = 0.0;
    double area_in_m2 = 0.0;
    double area_out_m2 = 0.0;
};

// Проверка корректности профиля.
// profile - профиль, который нужно проверить.
// min_allowed_area_m2 - минимально допустимая площадь, м^2.
void validate_profile(const AreaProfile& profile,
                      double min_allowed_area_m2 = 1e-10);

// Возвращает длину профиля, м.
// profile - уже заданный профиль трубы.
double profile_length_m(const AreaProfile& profile);

// Линейно интерполирует площадь профиля в точке x.
// profile - профиль трубы.
// x_m - координата, в которой нужна площадь, м.
double area_at_linear(const AreaProfile& profile, double x_m);

AreaProfile make_profile_from_points(const std::vector<ProfilePoint>& points);

AreaProfile make_profile_from_xy(const std::vector<double>& x_m,
                                 const std::vector<double>& area_m2);

AreaProfile make_linear_profile(double length_m,
                                double area_left_m2,
                                double area_right_m2);

AreaProfile make_profile_from_areas_uniform(
    double length_m,
    const std::vector<double>& area_samples_m2);

// Строит равномерную цилиндрическую аппроксимацию профиля.
// profile - исходный профиль.
// section_count - число цилиндрических секций.
// default_beta_loss_np_per_m - одинаковый коэффициент потерь для всех секций.
std::vector<CylinderSection> build_cylinders_uniform(
    const AreaProfile& profile,
    std::size_t section_count,
    double default_beta_loss_np_per_m = 0.0);

std::vector<ConeSection> build_cones_uniform(const AreaProfile& profile,
                                             std::size_t section_count);

double radius_from_area(double area_m2);

// Создает простой трехточечный профиль.
// Удобно для тестов: слева одна площадь, в середине другая, справа третья.
// length_m - длина трубы, м.
// area_left_m2 - площадь на входе, м^2.
// area_middle_m2 - площадь в середине, м^2.
// area_right_m2 - площадь на выходе, м^2.
AreaProfile make_three_point_profile(double length_m,
                                     double area_left_m2,
                                     double area_middle_m2,
                                     double area_right_m2);

}  // namespace vt_simple

#endif  // VT_GEOMETRY_SIMPLE_HPP
