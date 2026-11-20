"""Testes dos operadores de gradiente desenvolvidos por Fernando."""

import cv2
import numpy as np

from src.operadores import (
    KERNELS,
    aplicar_prewitt,
    aplicar_sobel,
    convolucao_manual,
)


def test_convolucao_manual_equivale_a_filter2d():
    """A implementação manual deve reproduzir a correlação do OpenCV."""
    gerador = np.random.default_rng(7)
    imagem = gerador.integers(0, 256, size=(13, 17), dtype=np.uint8)
    kernel = KERNELS["sobel"][0]

    manual = convolucao_manual(imagem, kernel)
    opencv = cv2.filter2D(imagem, cv2.CV_64F, kernel)

    assert manual.dtype == np.float64
    assert np.allclose(manual, opencv)


def test_soma_de_cada_kernel_e_zero():
    """Derivadas devem responder a mudanças, não ao brilho constante."""
    for kernel_x, kernel_y in KERNELS.values():
        assert np.isclose(kernel_x.sum(), 0.0)
        assert np.isclose(kernel_y.sum(), 0.0)


def test_imagem_constante_gera_gradiente_zero():
    """Uma região sem variação de brilho não tem bordas detectáveis."""
    imagem = np.full((9, 11), 73, dtype=np.uint8)

    for gx, gy in (aplicar_prewitt(imagem), aplicar_sobel(imagem)):
        assert np.allclose(gx, 0.0)
        assert np.allclose(gy, 0.0)


def test_borda_vertical_gera_gx_forte_e_gy_proximo_de_zero():
    """Uma mudança vertical no brilho varia em x, não em y."""
    imagem = np.zeros((9, 9), dtype=np.uint8)
    imagem[:, 4:] = 255
    gx, gy = aplicar_sobel(imagem)

    # O pixel à esquerda da transição recebe a resposta máxima do Sobel 3×3.
    assert abs(gx[4, 3]) > 0
    assert np.isclose(gy[4, 3], 0.0)
    assert np.max(np.abs(gx)) > 10 * np.max(np.abs(gy))
