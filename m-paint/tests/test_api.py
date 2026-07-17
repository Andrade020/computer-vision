import base64

from fastapi.testclient import TestClient

from server.app import app

client = TestClient(app)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_render_valid_latex():
    r = client.post("/api/render_latex",
                    json={"latex": r"\frac{\sqrt{x}}{2}", "height_px": 64})
    assert r.status_code == 200
    d = r.json()
    png = base64.b64decode(d["png_base64"])
    assert png[:8] == PNG_MAGIC
    assert d["width"] > 0
    assert d["height"] == 64


def test_render_custom_color_and_height():
    r = client.post("/api/render_latex",
                    json={"latex": "x^2", "height_px": 128,
                          "color": [200, 30, 30]})
    assert r.status_code == 200
    assert r.json()["height"] == 128


def test_bad_latex_falls_back_to_literal():
    r = client.post("/api/render_latex", json={"latex": r"\notacommand{{{"})
    assert r.status_code == 200
    png = base64.b64decode(r.json()["png_base64"])
    assert png[:8] == PNG_MAGIC


def test_empty_latex_is_422():
    r = client.post("/api/render_latex", json={"latex": ""})
    assert r.status_code == 422


def test_height_out_of_range_is_422():
    r = client.post("/api/render_latex", json={"latex": "x", "height_px": 9000})
    assert r.status_code == 422


def test_index_is_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "MathBoard" in r.text
