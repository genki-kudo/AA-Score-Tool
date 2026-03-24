#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// 入力:
//   coords_prot       : [Np,3]
//   q_prot            : [Np]
//   res_id_prot       : [Np] 残基ID (0..max_res, <0 は無視)
//   is_sidechain_prot : [Np] 1=sidechain, 0=main-chain
//   coords_lig        : [Nl,3]
//   q_lig             : [Nl]
// 出力 (tuple):
//   total,
//   per_res_side_same, per_res_side_opp,
//   per_res_main_same, per_res_main_opp
std::tuple<
    float,
    py::array_t<float>,  // per_res_side_same
    py::array_t<float>,  // per_res_side_opp
    py::array_t<float>,  // per_res_main_same
    py::array_t<float>   // per_res_main_opp
>
electrostatic_sum(
    py::array_t<float, py::array::c_style | py::array::forcecast> coords_prot,
    py::array_t<float, py::array::c_style | py::array::forcecast> q_prot,
    py::array_t<int,   py::array::c_style | py::array::forcecast> res_id_prot,
    py::array_t<int,   py::array::c_style | py::array::forcecast> is_sidechain_prot,
    py::array_t<float, py::array::c_style | py::array::forcecast> coords_lig,
    py::array_t<float, py::array::c_style | py::array::forcecast> q_lig,
    float r_cut,
    float ke = 1.0f
)
{
    auto P   = coords_prot.unchecked<2>();
    auto Qp  = q_prot.unchecked<1>();
    auto Rp  = res_id_prot.unchecked<1>();
    auto Sp  = is_sidechain_prot.unchecked<1>();  // 1 = side, 0 = main
    auto L   = coords_lig.unchecked<2>();
    auto Ql  = q_lig.unchecked<1>();

    const ssize_t Np = P.shape(0), Nl = L.shape(0);

    int max_res = 0;
    for (ssize_t i = 0; i < Np; ++i) {
        const int ri = Rp(i);
        if (ri > max_res)
            max_res = ri;
    }

    py::array_t<float> per_res_side_same(max_res + 1);
    py::array_t<float> per_res_side_opp (max_res + 1);
    py::array_t<float> per_res_main_same(max_res + 1);
    py::array_t<float> per_res_main_opp (max_res + 1);

    auto R_side_same = per_res_side_same.mutable_unchecked<1>();
    auto R_side_opp  = per_res_side_opp.mutable_unchecked<1>();
    auto R_main_same = per_res_main_same.mutable_unchecked<1>();
    auto R_main_opp  = per_res_main_opp.mutable_unchecked<1>();

    for (int r = 0; r <= max_res; ++r) {
        R_side_same(r) = 0.0f;
        R_side_opp(r)  = 0.0f;
        R_main_same(r) = 0.0f;
        R_main_opp(r)  = 0.0f;
    }

    float total = 0.0f;
    const float rc2 = r_cut * r_cut;

#pragma omp parallel
    {
        float total_local = 0.0f;
        std::vector<float> side_same_local(max_res + 1, 0.0f);
        std::vector<float> side_opp_local (max_res + 1, 0.0f);
        std::vector<float> main_same_local(max_res + 1, 0.0f);
        std::vector<float> main_opp_local (max_res + 1, 0.0f);

#pragma omp for schedule(static)
        for (ssize_t i = 0; i < Np; ++i) {
            const int ri = Rp(i);
            if (ri < 0 || ri > max_res)
                continue;

            const bool is_side = (Sp(i) != 0);

            const float xi = P(i, 0);
            const float yi = P(i, 1);
            const float zi = P(i, 2);
            const float qi = Qp(i);

            for (ssize_t j = 0; j < Nl; ++j) {
                const float dx = xi - L(j, 0);
                const float dy = yi - L(j, 1);
                const float dz = zi - L(j, 2);
                const float r2 = dx * dx + dy * dy + dz * dz;

                if (r2 <= 1e-12f || r2 > rc2)
                    continue;

                const float r = std::sqrt(r2);
                const float e = ke * qi * Ql(j) / r;

                total_local += e;

                if (is_side) {
                    if (e > 0.0f) {
                        side_same_local[ri] += e;
                    } else {
                        side_opp_local[ri]  += e;
                    }
                } else {
                    if (e > 0.0f) {
                        main_same_local[ri] += e;
                    } else {
                        main_opp_local[ri]  += e;
                    }
                }
            }
        }

#pragma omp critical
        {
            total += total_local;
            for (int r = 0; r <= max_res; ++r) {
                R_side_same(r) += side_same_local[r];
                R_side_opp(r)  += side_opp_local[r];
                R_main_same(r) += main_same_local[r];
                R_main_opp(r)  += main_opp_local[r];
            }
        }
    }

    return std::make_tuple(
        total,
        per_res_side_same,
        per_res_side_opp,
        per_res_main_same,
        per_res_main_opp
    );
}

PYBIND11_MODULE(fast_elec, m) {
    m.doc() = "Fast electrostatics sum (q_i q_j / r) with cutoff";

    m.def(
        "electrostatic_sum",
        &electrostatic_sum,
        py::arg("coords_prot"),
        py::arg("q_prot"),
        py::arg("res_id_prot"),
        py::arg("is_sidechain_prot"),
        py::arg("coords_lig"),
        py::arg("q_lig"),
        py::arg("r_cut"),
        py::arg("ke") = 1.0f
    );
}
