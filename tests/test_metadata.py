from anomaly_detection.metadata import load_metadata, save_metadata


def test_metadata_round_trip(tmp_path):
    path = tmp_path / "metadata.json"
    expected = {"threshold": 0.0123, "image_size": [128, 128]}

    save_metadata(path, expected)

    assert load_metadata(path) == expected
