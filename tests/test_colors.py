from clima_mollendo.colors import Scale


def test_scale_endpoints_and_middle():
    s = Scale(0.0, 1.0, 2.0)
    assert s.hex(0.0) == "#2ecc71"
    assert s.hex(1.0) == "#f1c40f"
    assert s.hex(2.0) == "#e74c3c"
    assert s.hex(5.0) == "#e74c3c"


def test_scale_none_and_labels():
    s = Scale(0.5, 1.5, 2.5)
    assert s.hex(None) == "#ffffff"
    assert s.css(None) == ""
    assert s.label(0.4) == "chico"
    assert s.label(1.5) == "medio"
    assert s.label(3.0) == "grande"
