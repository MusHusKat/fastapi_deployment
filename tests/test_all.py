from . import model


def test_compute_model_metrics():
    y_hat = [1, 0, 1, 0]
    y_true = [1, 0, 1, 0]
    resp = model.compute_model_metrics(y_hat, y_true)
    assert resp == (1.0, 1.0, 1.0)

    y_hat = [0, 0, 1, 0]
    y_true = [1, 0, 1, 0]
    resp = model.compute_model_metrics(y_hat, y_true)
    assert resp == (0.5, 1.0, 0.6666666666666666)

    y_hat = [0, 1, 0, 1]
    y_true = [1, 0, 1, 0]
    resp = model.compute_model_metrics(y_hat, y_true)
    assert resp == (0.0, 0.0, 0.0)


def test_save_artifacts(mocker):
    pickle_mocker = mocker.patch.object(model, "pickle")

    model_artifact = {"name": "dummy"}
    model.save_model_artifacts("dummy_path", **model_artifact)

    pickle_mocker.dump.assert_called()


def test_inference():

    # dummy model
    class mock_object:
        def predict(self, x):
            return [i ** 2 for i in x]

    mock_model = mock_object()
    preds = model.inference(mock_model, [1, 2, 3, 4])

    assert preds == [1, 4, 9, 16]
