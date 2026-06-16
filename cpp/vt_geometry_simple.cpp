#include "vt_geometry_simple.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vt_simple {

void validate_profile(const AreaProfile& profile,
                      double min_allowed_area_m2) {
    // Для трубы нужно как минимум две точки: начало и конец.
    if (profile.points.size() < 2) {
        throw std::invalid_argument(
            "AreaProfile must contain at least two points.");
    }

    // Проверяем каждую точку профиля.
    for (std::size_t i = 0; i < profile.points.size(); ++i) {
        const ProfilePoint& p = profile.points[i];

        // Площадь должна быть строго положительной.
        if (!(p.area_m2 > min_allowed_area_m2)) {
            throw std::invalid_argument(
                "All profile areas must be strictly positive.");
        }

        // Координаты вдоль оси должны строго возрастать.
        if (i > 0 && !(profile.points[i - 1].x_m < p.x_m)) {
            throw std::invalid_argument(
                "Profile x-coordinates must be strictly increasing.");
        }
    }
}

double profile_length_m(const AreaProfile& profile) {
    // Сначала убеждаемся, что профиль корректный.
    validate_profile(profile);

    // Длина равна разности последней и первой координаты.
    return profile.points.back().x_m - profile.points.front().x_m;
}

double area_at_linear(const AreaProfile& profile, double x_m) {
    // Проверяем входной профиль.
    validate_profile(profile);

    // Если запрос левее профиля, возвращаем левую крайнюю площадь.
    if (x_m <= profile.points.front().x_m) {
        return profile.points.front().area_m2;
    }

    // Если запрос правее профиля, возвращаем правую крайнюю площадь.
    if (x_m >= profile.points.back().x_m) {
        return profile.points.back().area_m2;
    }

    // Находим первую точку, лежащую строго правее x_m.
    const auto it = std::upper_bound(
        profile.points.begin(),
        profile.points.end(),
        x_m,
        [](double value, const ProfilePoint& p) { return value < p.x_m; });

    // Индексы соседних узлов, между которыми лежит x_m.
    const std::size_t right = static_cast<std::size_t>(it - profile.points.begin());
    const std::size_t left = right - 1;

    const ProfilePoint& p0 = profile.points[left];
    const ProfilePoint& p1 = profile.points[right];

    // Параметр линейной интерполяции на отрезке [p0, p1].
    const double t = (x_m - p0.x_m) / (p1.x_m - p0.x_m);

    // Линейная интерполяция площади.
    return p0.area_m2 * (1.0 - t) + p1.area_m2 * t;
}

AreaProfile make_profile_from_points(const std::vector<ProfilePoint>& points) {
    AreaProfile profile;
    profile.points = points;
    validate_profile(profile);
    return profile;
}

AreaProfile make_profile_from_xy(const std::vector<double>& x_m,
                                 const std::vector<double>& area_m2) {
    if (x_m.size() != area_m2.size()) {
        throw std::invalid_argument("x_m and area_m2 must have the same size.");
    }
    if (x_m.size() < 2) {
        throw std::invalid_argument("At least two profile points are required.");
    }

    std::vector<ProfilePoint> points;
    points.reserve(x_m.size());
    for (std::size_t i = 0; i < x_m.size(); ++i) {
        points.push_back(ProfilePoint{x_m[i], area_m2[i]});
    }

    return make_profile_from_points(points);
}

AreaProfile make_linear_profile(double length_m,
                                double area_left_m2,
                                double area_right_m2) {
    if (!(length_m > 0.0)) {
        throw std::invalid_argument("length_m must be positive.");
    }
    if (!(area_left_m2 > 0.0 && area_right_m2 > 0.0)) {
        throw std::invalid_argument("All profile areas must be positive.");
    }

    return make_profile_from_xy(
        std::vector<double>{0.0, length_m},
        std::vector<double>{area_left_m2, area_right_m2});
}

AreaProfile make_profile_from_areas_uniform(
    double length_m,
    const std::vector<double>& area_samples_m2) {
    if (!(length_m > 0.0)) {
        throw std::invalid_argument("length_m must be positive.");
    }
    if (area_samples_m2.size() < 2) {
        throw std::invalid_argument("At least two area samples are required.");
    }

    std::vector<ProfilePoint> points;
    points.reserve(area_samples_m2.size());

    const double dx =
        length_m / static_cast<double>(area_samples_m2.size() - 1);

    for (std::size_t i = 0; i < area_samples_m2.size(); ++i) {
        points.push_back(
            ProfilePoint{static_cast<double>(i) * dx, area_samples_m2[i]});
    }

    return make_profile_from_points(points);
}

std::vector<CylinderSection> build_cylinders_uniform(
    const AreaProfile& profile,
    std::size_t section_count,
    double default_beta_loss_np_per_m) {

    // Проверяем профиль перед аппроксимацией.
    validate_profile(profile);

    // Хотя бы одна секция обязательна.
    if (section_count == 0) {
        throw std::invalid_argument("section_count must be > 0.");
    }

    std::vector<CylinderSection> sections;
    sections.reserve(section_count);

    // Левая и правая границы всей трубы.
    const double x0 = profile.points.front().x_m;
    const double x1 = profile.points.back().x_m;

    // Шаг по оси x для равномерного разбиения трубы.
    const double dx = (x1 - x0) / static_cast<double>(section_count);

    for (std::size_t i = 0; i < section_count; ++i) {
        // Координаты текущей секции.
        const double left_x = x0 + static_cast<double>(i) * dx;
        const double right_x = left_x + dx;

        // Для цилиндра берем одну характерную площадь в центре секции.
        const double center_x = 0.5 * (left_x + right_x);

        CylinderSection sec;

        // Длина текущей секции.
        sec.length_m = dx;

        // Постоянная площадь цилиндра равна площади профиля в центре секции.
        sec.area_m2 = area_at_linear(profile, center_x);

        // Записываем коэффициент потерь в секцию.
        sec.beta_loss_np_per_m = default_beta_loss_np_per_m;

        // Добавляем секцию в результат.
        sections.push_back(sec);
    }

    return sections;
}

std::vector<ConeSection> build_cones_uniform(const AreaProfile& profile,
                                             std::size_t section_count) {
    validate_profile(profile);

    if (section_count == 0) {
        throw std::invalid_argument("section_count must be > 0.");
    }

    std::vector<ConeSection> sections;
    sections.reserve(section_count);

    const double x0 = profile.points.front().x_m;
    const double x1 = profile.points.back().x_m;
    const double dx = (x1 - x0) / static_cast<double>(section_count);

    for (std::size_t i = 0; i < section_count; ++i) {
        const double left_x = x0 + static_cast<double>(i) * dx;
        const double right_x = left_x + dx;

        ConeSection sec;
        sec.length_m = dx;
        sec.area_in_m2 = area_at_linear(profile, left_x);
        sec.area_out_m2 = area_at_linear(profile, right_x);
        sections.push_back(sec);
    }

    return sections;
}

double radius_from_area(double area_m2) {
    if (!(area_m2 > 0.0)) {
        throw std::invalid_argument("area_m2 must be positive.");
    }

    return std::sqrt(area_m2 / kPi);
}

AreaProfile make_three_point_profile(double length_m,
                                     double area_left_m2,
                                     double area_middle_m2,
                                     double area_right_m2) {
    // Длина профиля должна быть положительной.
    if (!(length_m > 0.0)) {
        throw std::invalid_argument("length_m must be positive.");
    }

    // Все площади должны быть положительными.
    if (!(area_left_m2 > 0.0 && area_middle_m2 > 0.0 && area_right_m2 > 0.0)) {
        throw std::invalid_argument("All profile areas must be positive.");
    }

    AreaProfile profile;

    // Левая граница трубы.
    profile.points.push_back(ProfilePoint{0.0, area_left_m2});

    // Середина трубы.
    profile.points.push_back(ProfilePoint{0.5 * length_m, area_middle_m2});

    // Правая граница трубы.
    profile.points.push_back(ProfilePoint{length_m, area_right_m2});

    return profile;
}

}  // namespace vt_simple
