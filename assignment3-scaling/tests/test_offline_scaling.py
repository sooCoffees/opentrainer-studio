from cs336_scaling.offline.configs import make_isoflops_grid, make_model_spec
from cs336_scaling.offline.fit import (
    best_points_by_compute,
    fit_compute_scaling_law,
    load_isoflops_points,
)
from cs336_scaling.offline.flops import (
    estimate_transformer_params,
    estimate_training_flops,
)


def test_make_model_spec_tracks_target_scale():
    small = make_model_spec(10_000_000)
    medium = make_model_spec(30_000_000)

    assert small.d_model % small.num_heads == 0
    assert medium.d_model % medium.num_heads == 0
    assert estimate_transformer_params(small) < estimate_transformer_params(medium)
    assert 8_000_000 <= estimate_transformer_params(small) <= 14_000_000


def test_make_isoflops_grid_uses_requested_compute_scale():
    grid = make_isoflops_grid(
        parameter_targets=[10_000_000, 30_000_000], compute_budgets=[1e17]
    )

    assert len(grid) == 2
    for spec in grid:
        flops = estimate_training_flops(
            estimate_transformer_params(spec.model), spec.train_tokens
        )
        assert 0.8e17 <= flops <= 1.2e17


def test_fit_scaling_law_on_fixture():
    points = load_isoflops_points("data/isoflops_curves.json")
    frontier = best_points_by_compute(points)
    fit = fit_compute_scaling_law(frontier)

    assert len(frontier) > 3
    assert fit.exponent < 0
    assert fit.rmse < 0.05
    assert fit.predict(1e20) < fit.predict(1e19)
