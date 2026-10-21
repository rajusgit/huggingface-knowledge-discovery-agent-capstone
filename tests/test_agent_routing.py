import json

import numpy as np
import pytest

from agents import router
from src.config import ROOT


def fake_embed_factory(table):
    def embed(texts, model=None):
        return [table[t] for t in texts]
    return embed


def unit(i, n=8):
    v = np.zeros(n)
    v[i] = 1.0
    return v.tolist()


def tiny_router(threshold=0.5, margin=0.05):
    routes = {"learn": ["learn u"], "discover": ["disc u"]}
    table = {"learn u": unit(0), "disc u": unit(1), "q learn": unit(0),
             "q disc": unit(1), "q none": unit(5), "q both": (np.array(unit(0)) + np.array(unit(1))).tolist()}
    return router.SemanticRouter(routes, fake_embed_factory(table), threshold=threshold, margin=margin)


def test_picks_nearest_route():
    assert tiny_router().classify("q learn").routes == ["learn"]
    assert tiny_router().classify("q disc").routes == ["discover"]


def test_below_threshold_is_out_of_scope():
    r = tiny_router().classify("q none")
    assert r.routes == [] and r.primary == "out_of_scope"


def test_two_close_routes_are_combined():
    r = tiny_router().classify("q both")
    assert set(r.routes) == {"learn", "discover"} and r.combined


def test_branch_mapping():
    assert router.branches(["learn"]) == (True, False)
    assert router.branches(["discover"]) == (False, True)
    assert router.branches(["compare", "evaluate"]) == (False, True)
    assert router.branches(["learn", "discover"]) == (True, True)
    assert router.branches([]) == (False, False)


def _scores(name):
    cases = json.loads((ROOT / "tests" / "fixtures" / name).read_text())
    r = router.load_router()
    intent = branch = 0
    for c in cases:
        primary = r.classify(c["q"]).primary
        intent += primary == c["expect"]
        branch += router.branches([primary]) == router.branches([c["expect"]])
    return intent / len(cases), branch / len(cases)


@pytest.mark.parametrize("fixture", ["route_eval.json", "route_test.json"])
def test_routing_accuracy(fixture):
    """Real encoder + configured routes.

    route_eval.json was used to tune routes.yaml; route_test.json was written afterwards and
    never tuned on. Branch (docs vs live tools) is what the graph acts on: >= 95%.
    Fine-grained intent (learn/discover/compare/evaluate) with MiniLM measured ~87-88%, below
    the plan's 95% target -- this floor guards against regressions, not the target.
    """
    intent, branch = _scores(fixture)
    assert branch >= 0.95, f"branch accuracy {branch:.0%}"
    assert intent >= 0.85, f"intent accuracy {intent:.0%}"
