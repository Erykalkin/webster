from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
CPP_DIR = ROOT / "cpp"
BIN_DIR = ROOT / "bin"
GENERATED_MAIN = CPP_DIR / "_generated_vt_cylinder_tlm_main.cpp"


def _platform_bin_dir() -> Path:
    system = platform.system().lower()
    if system.startswith("windows"):
        return BIN_DIR / "windows"
    if system == "darwin":
        return BIN_DIR / "macos"
    return BIN_DIR / "linux"


def _platform_executable_name(stem: str) -> str:
    if platform.system().lower().startswith("windows"):
        return f"{stem}.exe"
    return stem


DEFAULT_OUTPUT_NAME = _platform_executable_name("vt_cylinder_tlm_solver_demo")


def generate_main_cpp(
    points: int,
    sections: int,
    f_min_hz: float,
    f_max_hz: float,
) -> Path:
    source = f"""#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "vt_cylinder_tlm_solver.hpp"
#include "vt_geometry_simple.hpp"

int main() {{
    vt_simple::AcousticConstants acoustics;
    acoustics.rho_kg_m3 = 1.225;
    acoustics.c_m_s = 343.0;

    const vt_simple::AreaProfile profile = vt_simple::make_three_point_profile(
        0.17,
        3.0e-4,
        8.0e-5,
        4.0e-4
    );

    const std::vector<double> freqs = vt_simple::make_uniform_frequency_grid(
        {f_min_hz},
        {f_max_hz},
        {points}
    );

    const auto spectrum = vt_simple::solve_transfer_function_cylinder_tlm(
        profile,
        {sections},
        freqs,
        acoustics,
        0.0
    );

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "f_Hz  |  |H(f)|\\n";
    std::cout << "----------------\\n";

    for (const auto& sample : spectrum) {{
        std::cout << std::setw(6) << sample.frequency_hz << "  |  "
                  << std::setw(8) << std::abs(sample.transfer) << "\\n";
    }}

    return 0;
}}
"""
    GENERATED_MAIN.parent.mkdir(parents=True, exist_ok=True)
    GENERATED_MAIN.write_text(source, encoding="utf-8")
    return GENERATED_MAIN


def build_command(
    output_path: Path,
    points: int,
    sections: int,
    f_min_hz: float,
    f_max_hz: float,
) -> list[str]:
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError(
            "Не найден g++. Установите MinGW-w64 или добавьте g++ в PATH."
        )

    main_cpp = generate_main_cpp(points, sections, f_min_hz, f_max_hz)

    sources = [
        main_cpp,
        CPP_DIR / "vt_cylinder_tlm_solver.cpp",
        CPP_DIR / "vt_geometry_simple.cpp",
    ]

    return [
        compiler,
        "-std=c++17",
        "-O2",
        "-I",
        str(ROOT),
        *map(str, sources),
        "-o",
        str(output_path),
    ]


def compile_binary(
    output_path: Path,
    points: int,
    sections: int,
    f_min_hz: float,
    f_max_hz: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(output_path, points, sections, f_min_hz, f_max_hz)
    print("Сборка:")
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=ROOT)


def run_binary(output_path: Path) -> int:
    print("\nЗапуск:\n")
    completed = subprocess.run([str(output_path)], cwd=ROOT)
    return completed.returncode


def run_binary_capture(output_path: Path) -> str:
    completed = subprocess.run(
        [str(output_path)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def parse_solver_output(stdout: str) -> tuple[list[float], list[float]]:
    frequencies_hz: list[float] = []
    transfer_abs: list[float] = []

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or "|" not in line or line.startswith("f_Hz") or set(line) == {"-"}:
            continue

        left, right = line.split("|", maxsplit=1)
        frequencies_hz.append(float(left.strip()))
        transfer_abs.append(float(right.strip()))

    if not frequencies_hz:
        raise RuntimeError("Не удалось распарсить вывод solver.")

    return frequencies_hz, transfer_abs


def show_plot(frequencies_hz: list[float], transfer_abs: list[float]) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(frequencies_hz, transfer_abs, linewidth=1.5)
    plt.xlabel("f, Hz")
    plt.ylabel("|H(f)|")
    plt.title("Transfer Function: vt_cylinder_tlm_solver")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def solve(
    points: int = 32,
    sections: int = 20,
    f_min_hz: float = 50.0,
    f_max_hz: float = 5000.0,
    plot: bool = False,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> tuple[list[float], list[float]]:
    if points < 2:
        raise ValueError("points must be >= 2.")
    if sections < 1:
        raise ValueError("sections must be >= 1.")
    if f_max_hz <= f_min_hz:
        raise ValueError("f_max_hz must be greater than f_min_hz.")

    output_path = _platform_bin_dir() / output_name
    compile_binary(output_path, points, sections, f_min_hz, f_max_hz)
    stdout = run_binary_capture(output_path)
    frequencies_hz, transfer_abs = parse_solver_output(stdout)

    if plot:
        show_plot(frequencies_hz, transfer_abs)

    return frequencies_hz, transfer_abs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Компилирует и запускает пример для vt_cylinder_tlm_solver.cpp."
        )
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_NAME,
        help="Имя выходного exe-файла.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Только собрать exe, без запуска.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Построить график по выводу solver.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=32,
        help="Число точек по частоте.",
    )
    parser.add_argument(
        "--sections",
        type=int,
        default=20,
        help="Число цилиндрических секций трубы.",
    )
    parser.add_argument(
        "--f-min",
        type=float,
        default=50.0,
        help="Минимальная частота, Гц.",
    )
    parser.add_argument(
        "--f-max",
        type=float,
        default=5000.0,
        help="Максимальная частота, Гц.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _platform_bin_dir() / output_path

    if args.points < 2:
        print("Число точек --points должно быть >= 2.", file=sys.stderr)
        return 1
    if args.sections < 1:
        print("Число секций --sections должно быть >= 1.", file=sys.stderr)
        return 1
    if args.f_max <= args.f_min:
        print("--f-max должно быть больше --f-min.", file=sys.stderr)
        return 1

    try:
        compile_binary(
            output_path,
            points=args.points,
            sections=args.sections,
            f_min_hz=args.f_min,
            f_max_hz=args.f_max,
        )
    except subprocess.CalledProcessError as exc:
        print(f"\nОшибка компиляции, код возврата: {exc.returncode}", file=sys.stderr)
        return exc.returncode
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.build_only:
        print(f"\nГотово: {output_path}")
        return 0

    if args.plot:
        try:
            stdout = run_binary_capture(output_path)
            print("\nВывод solver:\n")
            print(stdout)
            frequencies_hz, transfer_abs = parse_solver_output(stdout)
            show_plot(frequencies_hz, transfer_abs)
            return 0
        except subprocess.CalledProcessError as exc:
            print(
                f"\nОшибка запуска solver, код возврата: {exc.returncode}",
                file=sys.stderr,
            )
            return exc.returncode
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    return run_binary(output_path)


if __name__ == "__main__":
    raise SystemExit(main())
