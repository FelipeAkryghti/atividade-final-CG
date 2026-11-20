"""Exemplo manual do Sobel em uma vizinhança com borda vertical.

O valor de intensidade muda de 0 para 255 na última coluna. O exemplo imprime
cada produto para que os cálculos possam ser conferidos no slide 10.
"""

import numpy as np

from src.operadores import KERNELS, direcao, magnitude


def main() -> None:
    """Calcula e imprime Gx, Gy, magnitude e ângulo no pixel central."""
    regiao = np.array(
        [[0, 0, 255], [0, 0, 255], [0, 0, 255]], dtype=np.float64
    )
    kernel_x, kernel_y = KERNELS["sobel"]

    produtos_x = regiao * kernel_x
    produtos_y = regiao * kernel_y
    gx = float(produtos_x.sum())
    gy = float(produtos_y.sum())
    mag = float(magnitude(np.array(gx), np.array(gy)))
    angulo = float(direcao(np.array(gx), np.array(gy)))

    print("Região 3x3 (a última coluna é mais clara):")
    print(regiao.astype(int))
    print("\nSobel Gx:")
    print(kernel_x.astype(int))
    print("Produtos região × Gx:")
    print(produtos_x.astype(int))
    print(f"Gx = soma dos produtos = {gx:.0f}")
    print("\nSobel Gy:")
    print(kernel_y.astype(int))
    print("Produtos região × Gy:")
    print(produtos_y.astype(int))
    print(f"Gy = soma dos produtos = {gy:.0f}")
    print(f"Magnitude = sqrt(Gx² + Gy²) = {mag:.0f}")
    print(f"Ângulo = arctan2(Gy, Gx) = {angulo:.4f} rad = {np.degrees(angulo):.1f}°")


if __name__ == "__main__":
    main()
