import numpy as np
import matplotlib.pyplot as plt


def tline_tube_along_x(S0, 
                       l, 
                       freq_grid, 
                       vloss=None,
                       wall_param=None
                       ):
    """
    Расчет передаточной функции трубы:
    - T_out(ω): на выходе (после всех сегментов)
    - T_x(ω, m): на стыке после m-го сегмента (усеченная труба)

    Параметры:
        S0 : 1D array, длина M
            Площади поперечных сечений.
        l : 1D array, длина M-1
            Длины сегментов.
        freq_grid : 1D array
            Частотная сетка в рад/с.
        vloss : float or None
            Параметр вязких потерь.
        wall_param : sequence of 3 floats or None
            Параметры стенок [R/Lw, 1/(Lw*Cw), w0].

    Возвращает:
        T_out : complex array, shape (Nω,)
            Передаточная функция на выходе.
        T_x : complex array, shape (Nω, M-1)
            Передаточные функции на стыках после каждого сегмента.
            T_x[k, m] соответствует частоте freq_grid[k]
            и координате x = sum(l[:m+1]).
    """
    S0 = np.asarray(S0, dtype=float)
    l = np.asarray(l, dtype=float)
    freq_grid = np.asarray(freq_grid, dtype=float)

    ro = 1.14e-3
    c = 35e3
    j = 1j
    tiny = 1e-15

    n_freq = len(freq_grid)
    n_seg = len(l)  # = len(S0) - 1

    # Излучение сейчас нулевое, как в исходном коде
    Zrad = np.zeros(n_freq, dtype=complex)
                  
    T_out = np.zeros(n_freq, dtype=complex)
    T_x = np.zeros((n_freq, n_seg), dtype=complex)

    for k, omega in enumerate(freq_grid):
        # потери
        if vloss is None:
            alpha = 0.0 + 0j
        else:
            alpha = np.sqrt(j * omega * vloss)

        # стенки
        if wall_param is None:
            hi = 0.0 + 0j
        else:
            R_over_Lw = wall_param[0]
            inv_LwCw = wall_param[1]
            w0 = wall_param[2]
            hi = j * omega * (w0 ** 2) / (
                inv_LwCw + j * omega * (j * omega + R_over_Lw)
            ) + alpha

        lambda_ = np.sqrt((alpha + j * omega) * (hi + j * omega)) / c

        num = np.sqrt(alpha + j * omega)
        den = np.sqrt(hi + j * omega)
        if abs(den) < tiny:
            dzeta = 1.0 + 0j
        else:
            dzeta = num / den

        # Для каждого сегмента заранее построим матрицу
        M_list = []
        for i in range(n_seg):
            H = np.array([[1.0, 0.0],
                          [0.0, 1.0 / S0[i]]], dtype=complex)

            G = np.array([
                [np.cosh(lambda_ * l[i]),
                 -ro * c * dzeta * np.sinh(lambda_ * l[i])],
                [-np.sinh(lambda_ * l[i]) / (ro * c * dzeta),
                 np.cosh(lambda_ * l[i])]
            ], dtype=complex)

            F = np.array([[1.0, 0.0],
                          [0.0, S0[i]]], dtype=complex)

            M = F @ G @ H
            M_list.append(M)

        # теперь считаем накопленные произведения и локальные T_m(ω)
        # M_pr после m-сегментов: M_m = M_m ... M_1
        M_pr = np.eye(2, dtype=complex)
        for m in range(n_seg):
            M_pr = M_list[m] @ M_pr
            A_m = M_pr[0, 0]
            C_m = M_pr[1, 0]

            # локальная передаточная функция усеченной трубы до этого стыка
            # сейчас Zrad = 0: T_m = 1 / A_m
            T_x[k, m] = 1.0 / (A_m - C_m * Zrad[k])

        # полная труба — после всех сегментов
        A = M_pr[0, 0]
        C = M_pr[1, 0]
        T_out[k] = 1.0 / (A - C * Zrad[k])

    return T_out, T_x


def simulate_tube(tube,
                  Fs=10_000, 
                  N=512,
                  vloss=None, 
                  wall_param=None,
                  do_plots=True
                  ):
    """
    tube:
        - либо функция без аргументов, которая возвращает (x_nodes, S, l)
        - либо кортеж/список (x_nodes, S, l)

    Fs : float
        Частота дискретизации (для частотной оси).
    N : int
        Количество шагов по частоте (0..Fs/2 -> N+1 точка).
    vloss, wall_param :
        Параметры потерь и стенок для tline_tube_along_x.
    """

    # 1) Получаем геометрию трубы
    if callable(tube):
        x_nodes, S, l = tube()
    else:
        # ожидаем, что tube = (x_nodes, S, l)
        x_nodes, S, l = tube

    x_nodes = np.asarray(x_nodes, dtype=float)
    S = np.asarray(S, dtype=float)
    l = np.asarray(l, dtype=float)

    n_nodes = len(x_nodes)
    n_seg = len(l)

    assert n_seg == n_nodes - 1, "len(l) должно быть len(x_nodes) - 1"
    assert len(S) == n_nodes, "len(S) должно совпадать с len(x_nodes)"

    # 2) Частотная сетка
    freq_hz = np.linspace(0.0, Fs / 2.0, N + 1)
    freq_grid = 2 * np.pi * freq_hz

    # 3) Считаем передаточную функцию трубы
    T_out, T_x = tline_tube_along_x(
        S, l, freq_grid,
        vloss=vloss,
        wall_param=wall_param
    )

    x_seg = np.cumsum(l)  # координаты стыков (длина = n_seg)

    if do_plots:
        # Профиль S(x)
        plt.figure()
        plt.plot(x_nodes, S, marker="o", linewidth=1)
        plt.xlabel("x")
        plt.ylabel("S(x)")
        plt.title("Профиль трубы")
        plt.grid(True)

        # АЧХ на выходе
        plt.figure()
        plt.plot(freq_hz, 20 * np.log10(np.abs(T_out) + 1e-12))
        plt.xlabel("f, Гц")
        plt.ylabel("|T(f)|, дБ")
        plt.title("АЧХ на выходе трубы")
        plt.grid(True)

        # Heatmap по длине
        amp_db = 20 * np.log10(np.abs(T_x) + 1e-12)
        plt.figure()
        plt.pcolormesh(x_seg, freq_hz, amp_db, shading="auto")
        plt.xlabel("x")
        plt.ylabel("f, Гц")
        plt.title("АЧХ(x, f)")
        plt.colorbar(label="Амплитуда, дБ")
        plt.tight_layout()

        plt.show()

    return freq_hz, T_out, x_seg, T_x
