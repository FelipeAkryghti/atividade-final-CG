"""Scharr e análise de parâmetros dos detectores de bordas por gradiente.

Responsável: Igor Domingues Pereira.

Os gradientes permanecem em float64. A normalização para 0..255 acontece
somente na visualização e antes de aplicar os limiares da figura 08.
"""

import cv2
import numpy as np


def aplicar_scharr(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (gx, gy) do Scharr 3x3, em float64 e no tamanho da imagem.

    img deve ser uma matriz 2D em tons de cinza. A profundidade CV_64F
    preserva tanto derivadas negativas quanto valores acima de 255.
    """
    imagem = np.asarray(img, dtype=np.float64)
    if imagem.ndim != 2 or imagem.size == 0:
        raise ValueError("A imagem deve ser uma matriz 2D não vazia em tons de cinza.")
    gx = cv2.Scharr(imagem, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(imagem, cv2.CV_64F, 0, 1)
    return gx, gy


def adicionar_ruido(img: np.ndarray, sigma: float = 20.0,
                    seed: int = 42) -> np.ndarray:
    """Adiciona ruído gaussiano de média zero e desvio padrão sigma.

    Retorna uma nova imagem uint8, limitada a 0..255. A mesma imagem,
    sigma e seed produzem o mesmo resultado. O gerador é local: não
    modifica a imagem original nem a semente global do NumPy.
    """
    imagem = np.asarray(img, dtype=np.float64)
    if imagem.ndim != 2 or imagem.size == 0:
        raise ValueError("A imagem deve ser uma matriz 2D não vazia em tons de cinza.")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma deve ser um número finito maior ou igual a zero.")
    gerador = np.random.default_rng(seed)
    ruido = gerador.normal(0.0, sigma, size=imagem.shape)
    return np.clip(imagem + ruido, 0, 255).astype(np.uint8)


def limiarizar(mag: np.ndarray, limiar: float = 80) -> np.ndarray:
    """Retorna uint8: 255 onde mag > limiar e 0 nos demais pixels.

    Não reescala mag internamente. Para usar limiares na escala 0..255
    com magnitudes brutas, passe normalizar(mag), como na figura 08.
    """
    return np.where(np.asarray(mag) > limiar, 255, 0).astype(np.uint8)


def figura_ksize(img: np.ndarray) -> str:
    """Salva 06_ksize.png: Sobel com ksize 1, 3, 5 e 7 lado a lado.

    Cada magnitude é normalizada separadamente, pois o ganho do Sobel
    aumenta com o tamanho do kernel. Scharr tem tamanho fixo 3x3.
    Retorna o caminho do arquivo salvo por mostrar_grade().
    """
    from src.operadores import aplicar_sobel, magnitude
    from src.utils import mostrar_grade, normalizar

    tamanhos = (1, 3, 5, 7)
    imagens = [normalizar(magnitude(*aplicar_sobel(img, ksize=k)))
               for k in tamanhos]
    titulos = [f"Sobel: ksize = {k}" for k in tamanhos]
    return mostrar_grade(imagens, titulos, colunas=4,
                         nome_saida="06_ksize.png")


def figura_blur_ruido(img: np.ndarray) -> str:
    """Salva 07_blur_ruido.png: ruído, Scharr sem e com GaussianBlur 5x5.

    A mesma imagem ruidosa (sigma=20, seed=42) alimenta os dois caminhos.
    A suavização é aplicada ANTES das derivadas. As duas magnitudes
    compartilham a escala de exibição para evidenciar a redução do ruído.
    """
    from src.operadores import magnitude
    from src.utils import mostrar_grade, normalizar

    ruidosa = adicionar_ruido(img)
    suavizada = cv2.GaussianBlur(ruidosa, (5, 5), sigmaX=0)
    sem_blur = magnitude(*aplicar_scharr(ruidosa))
    com_blur = magnitude(*aplicar_scharr(suavizada))

    # Normalizar a concatenação usa os mesmos mínimo e máximo nos dois mapas.
    mapas = normalizar(np.concatenate([sem_blur, com_blur], axis=1))
    sem_blur_vis, com_blur_vis = np.hsplit(mapas, 2)
    return mostrar_grade(
        [ruidosa, sem_blur_vis, com_blur_vis],
        ["Ruído gaussiano (σ = 20)", "Scharr sem blur",
         "GaussianBlur 5×5 → Scharr"],
        colunas=3, nome_saida="07_blur_ruido.png",
    )


def figura_limiar(img: np.ndarray) -> str:
    """Salva 08_limiar.png: magnitude Scharr e limiares 30, 80 e 150.

    Normaliza a magnitude uma única vez para 0..255 antes dos três
    limiares, permitindo comparar quais bordas sobrevivem a cada corte.
    """
    from src.operadores import magnitude
    from src.utils import mostrar_grade, normalizar

    mag = normalizar(magnitude(*aplicar_scharr(img)))
    limiares = (30, 80, 150)
    imagens = [mag] + [limiarizar(mag, valor) for valor in limiares]
    titulos = ["Magnitude Scharr (0..255)"] + [
        f"Limiar = {valor}" for valor in limiares
    ]
    return mostrar_grade(imagens, titulos, colunas=4,
                         nome_saida="08_limiar.png")


def comparar_rotacao() -> dict[str, float]:
    """Compara o erro angular de Sobel 3x3 e Scharr num círculo sintético.

    O disco claro tem raio 80 numa imagem 257x257. Sua borda suave segue
    I(r) = 127,5 * (1 - tanh((r - 80) / 2)), em float64; assim o ângulo
    real é conhecido e não há quantização uint8 ou degraus de um disco
    binário. O gradiente aponta para o centro, onde a intensidade cresce.

    Mede o erro absoluto na faixa |r - 80| <= 2, compartilhada pelos dois
    operadores. Usa diferença circular de ângulos, evitando erros de 360°
    na passagem entre -π e π. Para destacar direções oblíquas, também
    mede os pontos pelo menos 15° afastados dos eixos horizontal/vertical.

    Imprime e retorna médias em graus, nas chaves sobel, scharr,
    sobel_diagonais e scharr_diagonais. É uma demonstração controlada de
    isotropia, não uma garantia para toda imagem ruidosa ou rasterizada.
    """
    from src.operadores import aplicar_sobel, direcao

    y, x = np.indices((257, 257), dtype=np.float64)
    x -= 128.0
    y -= 128.0
    raio = np.hypot(x, y)
    circulo = 127.5 * (1.0 - np.tanh((raio - 80.0) / 2.0))
    angulo_real = np.arctan2(-y, -x)
    borda = np.abs(raio - 80.0) <= 2.0
    diagonais = borda & (np.abs(np.sin(2.0 * angulo_real)) >= 0.5)

    medias = {}
    for nome, operador in (("sobel", aplicar_sobel), ("scharr", aplicar_scharr)):
        gx, gy = operador(circulo)
        diferenca = direcao(gx, gy) - angulo_real
        erro = np.degrees(np.abs(np.arctan2(np.sin(diferenca),
                                           np.cos(diferenca))))
        medias[nome] = float(erro[borda].mean())
        medias[f"{nome}_diagonais"] = float(erro[diagonais].mean())

    print("Erro angular médio do gradiente em relação à direção real (graus):")
    print(f"Sobel 3x3: {medias['sobel']:.6f}° na borda; "
          f"{medias['sobel_diagonais']:.6f}° nas diagonais")
    print(f"Scharr 3x3: {medias['scharr']:.6f}° na borda; "
          f"{medias['scharr_diagonais']:.6f}° nas diagonais")
    return medias
