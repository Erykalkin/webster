range_library = {
    "cylinder": {
        "length_m": 1.0,
        "area_m2": (1.0e-4, 3.0e-2),
    },

    "conical": {
        "length_m": 1.0,
        "area_in_m2": (1.0e-4, 8.0e-3),
        "area_out_m2": (2.0e-4, 5.0e-2),
    },

    "three_point": {
        "length_m": 1.0,
        "area_left_m2": (1.0e-4, 8.0e-3),
        "area_middle_m2": (2.0e-4, 4.0e-2),
        "area_right_m2": (1.0e-4, 2.0e-2),
    },

    # "tube_with_hole": {
    #     "length_m": 1.0,
    #     "base_width_m": (0.015, 0.25),
    #     "random": True,
    # },

    "random_smooth": {
        "length_m": 1.0,
        "dx_m": (0.01, 0.05),
        "area0_m2": (1.5e-4, 2.0e-2),
        "amp": (0.1, 0.45),
        "n_harmonics": (3, 6),
    },

    "random_piecewise": {
        "length_m": 1.0,
        "mean_width_m": (0.015, 0.25),
        "section_count": (4, 12),
        "width_spread": (0.05, 0.4),
    },
}


hole_library = {
    "tube_with_hole": {
        "length_m": 1.0,
        "base_width_m": (0.015, 0.25),
        "random": True,
    }
}


length_range_library = {
    "cylinder": {
        "length_m": (0.5, 1.5),
        "area_m2": (1.0e-4, 3.0e-2),
    },

    "conical": {
        "length_m": (0.5, 1.5),
        "area_in_m2": (1.0e-4, 8.0e-3),
        "area_out_m2": (2.0e-4, 5.0e-2),
    },

    "three_point": {
        "length_m": (0.5, 1.5),
        "area_left_m2": (1.0e-4, 8.0e-3),
        "area_middle_m2": (2.0e-4, 4.0e-2),
        "area_right_m2": (1.0e-4, 2.0e-2),
    },

    "tube_with_hole": {
        "length_m": (0.5, 1.5),
        "base_width_m": (0.015, 0.25),
        "random": True,
    },

    "random_smooth": {
        "length_m": (0.5, 1.5),
        "dx_m": (0.01, 0.05),
        "area0_m2": (1.5e-4, 2.0e-2),
        "amp": (0.1, 0.45),
        "n_harmonics": (3, 6),
    },

    "random_piecewise": {
        "length_m": (0.5, 1.5),
        "mean_width_m": (0.015, 0.25),
        "section_count": (4, 12),
        "width_spread": (0.05, 0.4),
    },
}
