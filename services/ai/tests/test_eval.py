from rag.eval import serialize_evaluation_result


class FakeEvaluationResult:
    scores = [
        {
            "faithfulness": 0.75,
            "answer_relevancy": float("nan"),
        },
        {
            "faithfulness": None,
            "answer_relevancy": 0.5,
        },
    ]


def test_serialize_evaluation_result_normalizes_ragas_scores():
    assert serialize_evaluation_result(FakeEvaluationResult()) == {
        "faithfulness": [0.75, 0.0],
        "answer_relevancy": [0.0, 0.5],
    }
