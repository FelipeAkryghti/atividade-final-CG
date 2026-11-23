"""Gera as 9 figuras de imagens/saida/ de uma só vez (autor: Victor).

Uso:  python gerar_figuras.py [caminho_da_imagem]
Cada função chamada aqui pertence a um colega; este script só integra.
"""
import sys

import matplotlib

matplotlib.use("Agg")  # sem janelas: o script apenas grava arquivos

from src.operadores import figuras_sobel                      # Fernando: 02, 03, 04
from src.scharr_parametros import (figura_blur_ruido,         # Igor: 06, 07, 08
                                   figura_ksize, figura_limiar)
from src.utils import (carregar_cinza, figura_comparacao,     # Felipe: 05, 09
                       figura_direcao, salvar)


def main(caminho: str = "imagens/entrada/teste.png") -> None:
    """Carrega a imagem e chama, em ordem, as funções que geram as figuras 01 a 09."""
    img = carregar_cinza(caminho)          # uint8, tons de cinza
    caminhos = [salvar(img, "01_original.png")]            # 01 (Victor)
    caminhos += list(figuras_sobel(img))                   # 02, 03, 04
    caminhos.append(figura_comparacao(img))                # 05
    caminhos.append(figura_ksize(img))                     # 06
    caminhos.append(figura_blur_ruido(img))                # 07
    caminhos.append(figura_limiar(img))                    # 08
    caminhos.append(figura_direcao(img))                   # 09
    for c in caminhos:
        print("salvo:", c)


if __name__ == "__main__":
    main(*sys.argv[1:2])
