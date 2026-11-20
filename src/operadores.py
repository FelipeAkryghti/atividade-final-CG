"""Operadores de gradiente Prewitt e Sobel e operações auxiliares.

Responsável: Fernando Barros Greca.

Os operadores estimam as derivadas horizontal (Gx) e vertical (Gy) por
correlação com kernels 3×3. Os resultados são mantidos em float64 para
preservar valores negativos e evitar perder bordas de uma das polaridades.
"""

import cv2
import numpy as np


# Gy é a transposta de Gx. A orientação do kernel segue a convenção de
# correlação de cv2.filter2D (o kernel não é invertido).
KERNELS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "prewitt": (
        np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64),
        np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64),
    ),
    "sobel": (
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64),
        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64),
    ),
    "scharr": (
        np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float64),
        np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float64),
    ),
}


def convolucao_manual(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Aplica um kernel 2D com NumPy, reproduzindo cv2.filter2D.

    OpenCV realiza correlação (não inverte o kernel) e usa BORDER_DEFAULT,
    equivalente a reflexão sem repetir o pixel da borda. O preenchimento
    reflect do NumPy implementa essa mesma extensão para kernels ímpares.

    Na convolução matemática, giramos o kernel 180° antes de multiplicá-lo
    pela vizinhança; na correlação, usamos o kernel na orientação escrita.
    Para estes kernels de derivada, girar 180° troca o sinal: os pesos de
    lados opostos têm sinais opostos e o centro é zero. Por isso os valores
    de uma convolução com estes kernels são os da correlação com sinal
    invertido. A função segue filter2D e, portanto, calcula correlação.

    Parâmetros:
        img: imagem bidimensional, em tons de cinza.
        kernel: matriz bidimensional de dimensões ímpares.

    Retorna:
        Resultado float64, com o mesmo tamanho da imagem.
    """
    imagem = np.asarray(img, dtype=np.float64)
    mascara = np.asarray(kernel, dtype=np.float64)
    if imagem.ndim != 2:
        raise ValueError("A imagem deve ser uma matriz 2D em tons de cinza.")
    if mascara.ndim != 2 or any(tamanho % 2 == 0 for tamanho in mascara.shape):
        raise ValueError("O kernel deve ser uma matriz 2D com dimensões ímpares.")
    if imagem.size == 0 or mascara.size == 0:
        raise ValueError("A imagem e o kernel não podem estar vazios.")

    raio_y, raio_x = mascara.shape[0] // 2, mascara.shape[1] // 2
    # BORDER_REFLECT_101 espelha a imagem sem repetir a amostra da extremidade.
    ampliada = np.pad(imagem, ((raio_y, raio_y), (raio_x, raio_x)), mode="reflect")
    saida = np.zeros(imagem.shape, dtype=np.float64)
    for y in range(mascara.shape[0]):
        for x in range(mascara.shape[1]):
            # A multiplicação direta mantém a orientação usada por filter2D.
            saida += mascara[y, x] * ampliada[
                y:y + imagem.shape[0], x:x + imagem.shape[1]
            ]
    return saida


def aplicar_prewitt(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcula Gx e Gy da imagem usando os kernels Prewitt.

    Parâmetros:
        img: imagem 2D em tons de cinza.

    Retorna:
        Tupla (gx, gy) de matrizes float64, na mesma dimensão da imagem.
    """
    imagem = np.asarray(img)
    if imagem.ndim != 2:
        raise ValueError("A imagem deve ser uma matriz 2D em tons de cinza.")
    kernel_x, kernel_y = KERNELS["prewitt"]
    gx = cv2.filter2D(imagem, cv2.CV_64F, kernel_x)
    gy = cv2.filter2D(imagem, cv2.CV_64F, kernel_y)
    return gx, gy


def figuras_sobel(img: np.ndarray) -> tuple[str, str, str]:
    """Gera e salva as figuras Gx, Gy e magnitude do Sobel.

    Os mapas direcionais são normalizados individualmente para visualização
    em tons de cinza. Os gradientes usados no cálculo continuam em float64.

    Parâmetros:
        img: imagem 2D em tons de cinza.

    Retorna:
        Caminhos de 02_sobel_gx.png, 03_sobel_gy.png e
        04_sobel_magnitude.png, nessa ordem.
    """
    # Import local evita acoplar o uso dos operadores às rotinas de figura.
    from src.utils import normalizar, salvar

    gx, gy = aplicar_sobel(img)
    mag = magnitude(gx, gy)
    return (
        salvar(normalizar(gx), "02_sobel_gx.png"),
        salvar(normalizar(gy), "03_sobel_gy.png"),
        salvar(normalizar(mag), "04_sobel_magnitude.png"),
    )


def aplicar_sobel(img: np.ndarray, ksize: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Calcula Gx e Gy com cv2.Sobel e profundidade float64.

    Parâmetros:
        img: imagem 2D em tons de cinza.
        ksize: tamanho do kernel Sobel; OpenCV aceita 1, 3, 5 ou 7.

    Retorna:
        Tupla (gx, gy) de matrizes float64, na mesma dimensão da imagem.
    """
    imagem = np.asarray(img)
    if imagem.ndim != 2:
        raise ValueError("A imagem deve ser uma matriz 2D em tons de cinza.")
    if ksize not in (1, 3, 5, 7):
        raise ValueError("ksize deve ser 1, 3, 5 ou 7 para o operador Sobel.")
    gx = cv2.Sobel(imagem, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(imagem, cv2.CV_64F, 0, 1, ksize=ksize)
    return gx, gy


def magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Calcula a força do gradiente: sqrt(Gx² + Gy²), em float64.

    Parâmetros:
        gx: derivada horizontal.
        gy: derivada vertical, com o mesmo formato de gx.

    Retorna:
        Matriz float64 com a magnitude em cada pixel.
    """
    gradiente_x = np.asarray(gx, dtype=np.float64)
    gradiente_y = np.asarray(gy, dtype=np.float64)
    if gradiente_x.shape != gradiente_y.shape:
        raise ValueError("gx e gy precisam ter o mesmo formato.")
    return np.hypot(gradiente_x, gradiente_y)


def direcao(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Calcula a orientação do gradiente em radianos com arctan2(Gy, Gx).

    Parâmetros:
        gx: derivada horizontal.
        gy: derivada vertical, com o mesmo formato de gx.

    Retorna:
        Matriz float64 de ângulos no intervalo [-π, π].
    """
    gradiente_x = np.asarray(gx, dtype=np.float64)
    gradiente_y = np.asarray(gy, dtype=np.float64)
    if gradiente_x.shape != gradiente_y.shape:
        raise ValueError("gx e gy precisam ter o mesmo formato.")
    return np.arctan2(gradiente_y, gradiente_x)
