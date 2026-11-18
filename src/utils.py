"""
Funções utilitárias do projeto (Grupo 05 | Tema 03 | Detectores de Bordas por Gradiente).

Responsável: Felipe Caldeira Akryghti.

Este módulo reúne o que os outros módulos usam o tempo todo:
    - carregar_cinza : lê a imagem do disco já em tons de cinza (uint8)
    - normalizar     : converte gradientes float64 (com negativos) em uint8 0..255
    - salvar         : grava resultados em imagens/saida/
    - mostrar_grade  : monta uma grade de imagens com matplotlib
    - mapa_direcao   : pinta a direção do gradiente (matiz) e a força (brilho)

E gera duas figuras da apresentação:
    - figura_comparacao -> imagens/saida/05_comparacao_operadores.png
    - figura_direcao    -> imagens/saida/09_direcao.png
"""

import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Raiz do repositório = pasta acima de src/.
# Usamos o caminho deste arquivo (e não a pasta atual do terminal) para que
# o código funcione igual quando chamado do notebook (notebooks/), do
# Streamlit (app/) ou da raiz do repositório.
RAIZ = Path(__file__).resolve().parent.parent
PASTA_SAIDA = RAIZ / "imagens" / "saida"


# ---------------------------------------------------------------------------
# Funções do contrato
# ---------------------------------------------------------------------------

def carregar_cinza(caminho: str) -> np.ndarray:
    """Carrega uma imagem do disco em tons de cinza.

    Caminhos relativos são procurados primeiro a partir da pasta atual e,
    se não existirem ali, a partir da raiz do repositório. Assim
    "imagens/entrada/teste.png" funciona tanto na raiz quanto em notebooks/.

    Parâmetros:
        caminho: caminho do arquivo de imagem (png, jpg, ...).

    Retorna:
        Matriz 2D uint8 (valores de 0 a 255), um valor de brilho por pixel.

    Erros:
        FileNotFoundError: o arquivo não existe.
        ValueError: o arquivo existe, mas não é uma imagem válida.
    """
    arquivo = Path(caminho)
    if not arquivo.is_absolute() and not arquivo.exists():
        arquivo = RAIZ / arquivo

    if not arquivo.is_file():
        raise FileNotFoundError(
            f"Imagem não encontrada: '{caminho}'. Confira o nome do arquivo "
            f"e a pasta (exemplo: imagens/entrada/teste.png)."
        )

    # cv2.imread falha no Windows quando o caminho tem acentos
    # (ex.: "Área de Trabalho"). Por isso lemos os bytes com NumPy e
    # deixamos o OpenCV só decodificar o conteúdo.
    dados = np.fromfile(str(arquivo), dtype=np.uint8)
    imagem = None
    if dados.size > 0:
        # IMREAD_GRAYSCALE converte cor em cinza pela luminância:
        # Y = 0,299 R + 0,587 G + 0,114 B
        imagem = cv2.imdecode(dados, cv2.IMREAD_GRAYSCALE)

    if imagem is None:
        raise ValueError(
            f"O arquivo '{arquivo}' existe, mas não é uma imagem que o "
            f"OpenCV consiga ler."
        )
    return imagem


def normalizar(img: np.ndarray) -> np.ndarray:
    """Reescala os valores da imagem para 0..255 (min-max) e devolve uint8.

    Os gradientes saem em float64, com valores negativos e acima de 255,
    e não podem ser exibidos nem salvos diretamente. A normalização
    min-max leva o menor valor para 0 e o maior para 255:

        saida = (img - min) / (max - min) * 255

    Uma imagem constante (max == min, sem nenhuma borda) vira toda preta.

    Parâmetros:
        img: matriz de qualquer tipo numérico (uint8, float64, ...).

    Retorna:
        Matriz uint8 com o mesmo formato, valores entre 0 e 255.
    """
    # float64 primeiro: o OpenCV não aceita alguns tipos inteiros (ex.: int64).
    img = np.asarray(img, dtype=np.float64)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def salvar(img: np.ndarray, nome: str) -> str:
    """Salva uma imagem em imagens/saida/, criando a pasta se ela não existir.

    Se a imagem não for uint8 (por exemplo, um gradiente float64), ela é
    normalizada para 0..255 antes de gravar. Sem extensão no nome, usa .png.

    Parâmetros:
        img: imagem em tons de cinza (2D) ou colorida em BGR (3 canais).
        nome: nome do arquivo, por exemplo "04_sobel_magnitude.png".

    Retorna:
        O caminho completo do arquivo salvo, como texto.
    """
    caminho = _caminho_saida(nome)
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = normalizar(img)

    # cv2.imwrite também falha com acentos no caminho no Windows, então
    # codificamos a imagem em memória (imencode) e gravamos os bytes.
    try:
        ok, conteudo = cv2.imencode(caminho.suffix, img)
    except cv2.error as erro:
        raise ValueError(f"Não foi possível salvar '{nome}': {erro}") from erro
    if not ok:
        raise ValueError(f"Não foi possível codificar a imagem '{nome}'.")
    conteudo.tofile(str(caminho))
    return str(caminho)


def mostrar_grade(imagens: list, titulos: list, colunas: int = 3,
                  nome_saida: str | None = None):
    """Mostra várias imagens lado a lado em uma grade (matplotlib).

    Regras de exibição:
        - imagens 2D aparecem em tons de cinza, de 0 (preto) a 255 (branco);
        - imagens que não são uint8 (gradientes float64) são normalizadas
          antes, com normalizar(); em Gx e Gy o zero vira cinza médio;
        - imagens de 3 canais são tratadas como BGR (padrão do OpenCV) e
          convertidas para RGB, que é o que o matplotlib espera;
        - os eixos (réguas com números) são escondidos.

    Em notebooks (Jupyter/Colab) a figura aparece na célula. Em scripts sem
    janela (backend "Agg", como no gerar_figuras.py) ela só é salva.

    Parâmetros:
        imagens: lista de matrizes NumPy.
        titulos: lista de títulos, um para cada imagem.
        colunas: quantas imagens por linha (padrão 3).
        nome_saida: se informado, salva a figura em imagens/saida/<nome_saida>.

    Retorna:
        O caminho do arquivo salvo, ou None se nome_saida não foi informado.
    """
    if len(imagens) == 0:
        raise ValueError("A lista de imagens está vazia.")
    if len(imagens) != len(titulos):
        raise ValueError(
            f"São {len(imagens)} imagens e {len(titulos)} títulos; "
            f"as quantidades precisam ser iguais."
        )

    colunas = max(1, min(colunas, len(imagens)))
    linhas = math.ceil(len(imagens) / colunas)
    figura, eixos = plt.subplots(linhas, colunas,
                                 figsize=(4.5 * colunas, 4.5 * linhas),
                                 squeeze=False)

    for eixo in eixos.flat:
        eixo.axis("off")  # também esconde as células vazias da última linha

    for eixo, img, titulo in zip(eixos.flat, imagens, titulos):
        img = np.asarray(img)
        if img.dtype != np.uint8:
            img = normalizar(img)
        if img.ndim == 3:
            eixo.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            # vmin/vmax fixos: 0 é sempre preto e 255 sempre branco
            # (sem isso o matplotlib reescalaria cada imagem por conta própria).
            eixo.imshow(img, cmap="gray", vmin=0, vmax=255)
        eixo.set_title(titulo, fontsize=13)

    figura.tight_layout()

    caminho = None
    if nome_saida:
        caminho = str(_caminho_saida(nome_saida))
        figura.savefig(caminho, dpi=150, bbox_inches="tight", facecolor="white")
    _exibir_e_fechar(figura)
    return caminho


def mapa_direcao(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Gera uma imagem colorida da direção do gradiente.

    Usa o espaço de cores HSV:
        H (matiz)      = direção  theta = arctan2(Gy, Gx)
        S (saturação)  = 255 (cores puras)
        V (brilho)     = magnitude G = sqrt(Gx² + Gy²), normalizada para 0..255

    Assim, a cor diz para onde a intensidade cresce e o brilho diz quão
    forte é a borda. Regiões lisas (G perto de 0) ficam pretas.

    Convenção de ângulos (o eixo y da imagem aponta para BAIXO):
        0° -> direita (vermelho), 90° -> baixo, 180° -> esquerda (ciano),
        270° (= -90°) -> cima.

    Parâmetros:
        gx: derivada na direção x (float64).
        gy: derivada na direção y (float64), mesmo formato de gx.

    Retorna:
        Imagem BGR uint8 (formato do OpenCV), mesmo tamanho de gx.
    """
    gx = np.asarray(gx, dtype=np.float64)
    gy = np.asarray(gy, dtype=np.float64)
    if gx.shape != gy.shape:
        raise ValueError(f"gx {gx.shape} e gy {gy.shape} têm formatos diferentes.")

    # arctan2 devolve o ângulo em (-180°, 180°]; o resto da divisão por 360
    # leva para [0°, 360°), que é a volta completa do círculo de cores.
    angulo = np.degrees(np.arctan2(gy, gx)) % 360.0

    # No OpenCV, com uint8, a matiz vai de 0 a 179 (cada unidade = 2 graus).
    matiz = (np.round(angulo / 2.0) % 180).astype(np.uint8)
    saturacao = np.full(gx.shape, 255, dtype=np.uint8)
    brilho = normalizar(np.hypot(gx, gy))  # hypot = sqrt(gx² + gy²)

    hsv = cv2.merge([matiz, saturacao, brilho])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# Figuras da apresentação (05 e 09)
# ---------------------------------------------------------------------------

def figura_comparacao(img: np.ndarray) -> str:
    """Gera 05_comparacao_operadores.png: magnitude de Prewitt, Sobel e Scharr.

    Mostra lado a lado a imagem original e a magnitude G = sqrt(Gx² + Gy²)
    de cada operador. Cada magnitude é normalizada para 0..255 para ser
    exibida; o título mostra o valor máximo original, porque a escala dos
    operadores é diferente (soma dos pesos de um lado do kernel:
    Prewitt 3, Sobel 4, Scharr 16).

    Parâmetros:
        img: imagem em tons de cinza (uint8).

    Retorna:
        O caminho do arquivo salvo.
    """
    # Importados aqui dentro (e não no topo do arquivo) para evitar import
    # circular: os outros módulos também podem importar funções deste.
    from src.operadores import aplicar_prewitt, aplicar_sobel, magnitude
    from src.scharr_parametros import aplicar_scharr

    magnitudes = {
        "Prewitt": magnitude(*aplicar_prewitt(img)),
        "Sobel": magnitude(*aplicar_sobel(img)),
        "Scharr": magnitude(*aplicar_scharr(img)),
    }

    imagens = [img]
    titulos = ["Original"]
    for nome, mag in magnitudes.items():
        imagens.append(normalizar(mag))
        titulos.append(f"{nome}\nmáx = {mag.max():.0f}")

    return mostrar_grade(imagens, titulos, colunas=4,
                         nome_saida="05_comparacao_operadores.png")


def figura_direcao(img: np.ndarray) -> str:
    """Gera 09_direcao.png: mapa de direção do gradiente com legenda de cores.

    Calcula Gx e Gy com Sobel 3x3, pinta a direção com mapa_direcao() e
    desenha ao lado uma roda de cores que explica qual cor corresponde a
    cada ângulo.

    Parâmetros:
        img: imagem em tons de cinza (uint8).

    Retorna:
        O caminho do arquivo salvo.
    """
    from src.operadores import aplicar_sobel  # import local (ver figura_comparacao)

    gx, gy = aplicar_sobel(img)
    mapa = mapa_direcao(gx, gy)

    figura = plt.figure(figsize=(15, 5.4))
    grade = figura.add_gridspec(1, 3, width_ratios=[1, 1, 0.8], wspace=0.08)

    eixo_original = figura.add_subplot(grade[0])
    eixo_original.imshow(img, cmap="gray", vmin=0, vmax=255)
    eixo_original.set_title("Original", fontsize=13)
    eixo_original.axis("off")

    eixo_mapa = figura.add_subplot(grade[1])
    eixo_mapa.imshow(cv2.cvtColor(mapa, cv2.COLOR_BGR2RGB))
    eixo_mapa.set_title("Direção do gradiente (Sobel 3×3)", fontsize=13)
    eixo_mapa.axis("off")

    _desenhar_legenda_direcao(figura.add_subplot(grade[2]))

    caminho = str(_caminho_saida("09_direcao.png"))
    figura.savefig(caminho, dpi=150, bbox_inches="tight", facecolor="white")
    _exibir_e_fechar(figura)
    return caminho


# ---------------------------------------------------------------------------
# Funções auxiliares internas (o "_" no nome indica uso só neste arquivo)
# ---------------------------------------------------------------------------

def _caminho_saida(nome: str) -> Path:
    """Monta o caminho imagens/saida/<nome> e cria as pastas que faltarem."""
    caminho = PASTA_SAIDA / nome
    if caminho.suffix == "":
        caminho = caminho.with_suffix(".png")
    caminho.parent.mkdir(parents=True, exist_ok=True)
    return caminho


def _exibir_e_fechar(figura) -> None:
    """Mostra a figura quando há onde mostrar e depois libera a memória.

    No Jupyter/Colab, plt.show() desenha a figura na célula. Com o backend
    "Agg" (scripts e testes, sem janela) não há o que mostrar.
    """
    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(figura)


def _roda_de_cores(tamanho: int = 301) -> np.ndarray:
    """Cria a roda de cores da legenda usando a própria mapa_direcao().

    Em cada ponto do disco o "gradiente" aponta para fora do centro
    (gx = x, gy = y). Assim cada ângulo recebe exatamente a cor que
    mapa_direcao() usa na imagem real, e o brilho cresce do centro
    (magnitude 0, preto) para a borda (magnitude máxima).

    Retorna:
        Imagem RGBA (fora do disco fica transparente).
    """
    eixo = np.linspace(-1.0, 1.0, tamanho)
    x, y = np.meshgrid(eixo, eixo)  # y cresce para baixo, como nas linhas da imagem
    dentro = np.hypot(x, y) <= 1.0
    roda = mapa_direcao(np.where(dentro, x, 0.0), np.where(dentro, y, 0.0))
    alfa = np.where(dentro, 255, 0).astype(np.uint8)
    return np.dstack([cv2.cvtColor(roda, cv2.COLOR_BGR2RGB), alfa])


def _desenhar_legenda_direcao(eixo) -> None:
    """Desenha a roda de cores com os ângulos principais marcados."""
    # extent com y de cima (-1) para baixo (+1): mesmo sentido da imagem.
    eixo.imshow(_roda_de_cores(), extent=(-1, 1, 1, -1))
    eixo.set_xlim(-1.9, 1.9)
    eixo.set_ylim(2.35, -1.75)  # y invertido: valores maiores ficam embaixo
    eixo.set_aspect("equal")
    eixo.axis("off")

    estilo = {"fontsize": 11, "color": "#333333", "ha": "center", "va": "center"}
    eixo.text(1.42, 0, "0°\n→", **estilo)
    eixo.text(0, 1.3, "↓ 90°", **estilo)
    eixo.text(-1.42, 0, "180°\n←", **estilo)
    eixo.text(0, -1.3, "↑ 270° (−90°)", **estilo)

    eixo.set_title("Legenda das cores", fontsize=13)
    eixo.text(0, 1.95,
              "cor = direção θ = arctan2(Gy, Gx)\n"
              "brilho = magnitude |G|\n"
              "o gradiente aponta do escuro para o claro",
              fontsize=10, color="#333333", ha="center", va="center")


if __name__ == "__main__":
    # Uso (na raiz do repositório):  python -m src.utils
    # Gera as figuras 05 e 09 a partir de imagens/entrada/teste.png.
    plt.switch_backend("Agg")
    imagem = carregar_cinza("imagens/entrada/teste.png")
    print("Figura salva em:", figura_comparacao(imagem))
    print("Figura salva em:", figura_direcao(imagem))
