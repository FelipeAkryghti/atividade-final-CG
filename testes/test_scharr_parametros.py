"""Testes de Scharr e parâmetros (responsável: Igor Domingues Pereira).

Na raiz do repositório: python -m pytest testes/test_scharr_parametros.py -v
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src import operadores, utils  # noqa: E402
from src.scharr_parametros import (  # noqa: E402
    adicionar_ruido,
    aplicar_scharr,
    comparar_rotacao,
    figura_blur_ruido,
    figura_ksize,
    figura_limiar,
    limiarizar,
)


@pytest.fixture
def imagem():
    """Inclui transições claras/escuras, retas e curvas."""
    img = np.full((64, 64), 30, dtype=np.uint8)
    cv2.rectangle(img, (6, 8), (28, 54), 150, -1)
    cv2.circle(img, (46, 32), 13, 240, -1)
    return img


def test_aplicar_scharr_retorna_float64_e_preserva_sinais():
    img = np.zeros((9, 11), dtype=np.uint8)
    img[:, 3:8] = 255
    gx, gy = aplicar_scharr(img)

    assert gx.dtype == gy.dtype == np.float64
    assert gx.shape == gy.shape == img.shape
    assert gx[4, 2] == 4080.0
    assert gx[4, 7] == -4080.0
    np.testing.assert_allclose(gy, 0.0)


def test_aplicar_scharr_imagem_constante_nao_tem_bordas():
    gx, gy = aplicar_scharr(np.full((8, 10), 73, dtype=np.uint8))
    np.testing.assert_allclose(gx, 0.0)
    np.testing.assert_allclose(gy, 0.0)


def test_adicionar_ruido_e_reprodutivel_com_a_mesma_semente(imagem):
    original = imagem.copy()
    a = adicionar_ruido(imagem, sigma=20.0, seed=7)
    b = adicionar_ruido(imagem, sigma=20.0, seed=7)
    c = adicionar_ruido(imagem, sigma=20.0, seed=8)

    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(imagem, original)
    assert not np.array_equal(a, c)
    assert a.dtype == np.uint8
    assert a.shape == imagem.shape


def test_adicionar_ruido_limita_a_faixa_sem_estouro():
    img = np.full((100, 100), 128, dtype=np.uint8)
    ruidosa = adicionar_ruido(img, sigma=1000.0, seed=7)
    assert ruidosa.min() == 0
    assert ruidosa.max() == 255


def test_adicionar_ruido_sigma_zero_preserva_imagem(imagem):
    np.testing.assert_array_equal(adicionar_ruido(imagem, sigma=0), imagem)


@pytest.mark.parametrize("limiar", [30, 80, 150])
def test_limiarizar_retorna_apenas_zero_e_255(limiar):
    mag = np.array([[0.0, limiar - 0.5, limiar, limiar + 0.5, 255.0]])
    resultado = limiarizar(mag, limiar)

    assert resultado.dtype == np.uint8
    assert resultado.shape == mag.shape
    assert set(np.unique(resultado)) == {0, 255}
    np.testing.assert_array_equal(resultado, [[0, 0, 0, 255, 255]])


@pytest.mark.parametrize("funcao, nome", [
    (figura_ksize, "06_ksize.png"),
    (figura_blur_ruido, "07_blur_ruido.png"),
    (figura_limiar, "08_limiar.png"),
])
def test_figuras_geram_png_legivel(funcao, nome, imagem, tmp_path, monkeypatch):
    pasta = tmp_path / "saída"
    monkeypatch.setattr(utils, "PASTA_SAIDA", pasta)
    caminho = Path(funcao(imagem))

    assert caminho == pasta / nome
    resultado = utils.carregar_cinza(str(caminho))
    assert resultado.shape[1] > resultado.shape[0]
    assert resultado.std() > 0


def test_figura_ksize_usa_os_quatro_tamanhos(imagem, monkeypatch):
    tamanhos = []
    sobel_original = operadores.aplicar_sobel

    def registrar_sobel(img, ksize=3):
        tamanhos.append(ksize)
        return sobel_original(img, ksize=ksize)

    def conferir_grade(imagens, titulos, colunas, nome_saida):
        assert len(imagens) == len(titulos) == colunas == 4
        return nome_saida

    monkeypatch.setattr(operadores, "aplicar_sobel", registrar_sobel)
    monkeypatch.setattr(utils, "mostrar_grade", conferir_grade)
    figura_ksize(imagem)
    assert tamanhos == [1, 3, 5, 7]


def test_figura_blur_ruido_suaviza_antes_do_gradiente(imagem, monkeypatch):
    import src.scharr_parametros as modulo

    entradas = []
    scharr_original = aplicar_scharr

    def registrar_scharr(img):
        entradas.append(img.copy())
        return scharr_original(img)

    monkeypatch.setattr(modulo, "aplicar_scharr", registrar_scharr)
    monkeypatch.setattr(utils, "mostrar_grade", lambda *args, **kwargs: "figura")
    figura_blur_ruido(imagem)

    ruidosa = adicionar_ruido(imagem)
    assert len(entradas) == 2
    np.testing.assert_array_equal(entradas[0], ruidosa)
    np.testing.assert_array_equal(entradas[1],
                                  cv2.GaussianBlur(ruidosa, (5, 5), sigmaX=0))


def test_figura_limiar_usa_a_mesma_magnitude_normalizada(imagem, monkeypatch):
    def conferir_grade(imagens, titulos, colunas, nome_saida):
        assert len(imagens) == len(titulos) == colunas == 4
        mag, *mascaras = imagens
        assert mag.dtype == np.uint8
        for valor, mascara in zip((30, 80, 150), mascaras):
            np.testing.assert_array_equal(mascara > 0, mag > valor)
        return nome_saida

    monkeypatch.setattr(utils, "mostrar_grade", conferir_grade)
    assert figura_limiar(imagem) == "08_limiar.png"


def test_comparar_rotacao_scharr_tem_menor_erro_angular(capsys):
    medias = comparar_rotacao()

    assert 0 <= medias["scharr"] < medias["sobel"] < 1.0
    assert 0 <= medias["scharr_diagonais"] < medias["sobel_diagonais"] < 1.0
    texto = capsys.readouterr().out
    assert "Sobel 3x3" in texto
    assert "Scharr 3x3" in texto
    assert "diagonais" in texto
