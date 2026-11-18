"""
Testes automáticos de src/utils.py (responsável: Felipe Caldeira Akryghti).

Como rodar, na raiz do repositório:
    python -m pytest testes/test_utils.py -v

Os testes que salvam arquivos usam uma pasta temporária do pytest, então
não sujam imagens/saida/ do repositório.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem janelas durante os testes (precisa vir antes do src.utils)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src import utils  # noqa: E402
from src.utils import (  # noqa: E402
    carregar_cinza,
    mapa_direcao,
    mostrar_grade,
    normalizar,
    salvar,
)


@pytest.fixture
def pasta_saida(tmp_path, monkeypatch):
    """Troca imagens/saida/ por uma pasta temporária que ainda não existe."""
    pasta = tmp_path / "saida_teste"
    monkeypatch.setattr(utils, "PASTA_SAIDA", pasta)
    return pasta


def imagem_com_degrau() -> np.ndarray:
    """Imagem 20x20: metade esquerda preta (0), metade direita branca (255)."""
    img = np.zeros((20, 20), dtype=np.uint8)
    img[:, 10:] = 255
    return img


# ---------------------------------------------------------------- normalizar

def test_normalizar_retorna_uint8_entre_0_e_255():
    # Valores parecidos com um gradiente real: float64, negativos e acima de 255.
    gradiente = np.array([[-1020.0, -3.5], [0.0, 4080.0]])
    saida = normalizar(gradiente)

    assert saida.dtype == np.uint8
    assert saida.shape == gradiente.shape
    assert saida.min() == 0      # o menor valor vira preto
    assert saida.max() == 255    # o maior valor vira branco


def test_normalizar_mantem_a_ordem_dos_valores():
    valores = np.linspace(-50.0, 50.0, 100).reshape(10, 10)
    saida = normalizar(valores).ravel().astype(int)
    assert np.all(np.diff(saida) >= 0)  # quem era maior continua maior ou igual


def test_normalizar_imagem_constante_vira_preta():
    assert np.all(normalizar(np.full((5, 5), 7.0)) == 0)


# -------------------------------------------------------------------- salvar

def test_salvar_cria_a_pasta_e_o_arquivo(pasta_saida):
    assert not pasta_saida.exists()

    img = imagem_com_degrau()
    caminho = Path(salvar(img, "teste_salvar.png"))

    assert caminho.is_file()
    assert caminho.parent == pasta_saida
    assert np.array_equal(carregar_cinza(str(caminho)), img)  # PNG não perde nada


def test_salvar_converte_gradiente_float_para_uint8(pasta_saida):
    caminho = salvar(np.array([[-1.0, 0.0, 1.0]]), "gradiente.png")
    assert carregar_cinza(caminho).tolist() == [[0, 128, 255]]


# ------------------------------------------------------------ carregar_cinza

def test_carregar_cinza_caminho_invalido_da_erro_claro():
    with pytest.raises(FileNotFoundError, match="Imagem não encontrada.*nao_existe.png"):
        carregar_cinza("imagens/entrada/nao_existe.png")


def test_carregar_cinza_arquivo_que_nao_e_imagem(tmp_path):
    falso = tmp_path / "falso.png"
    falso.write_text("isto não é uma imagem", encoding="utf-8")
    with pytest.raises(ValueError, match="não é uma imagem"):
        carregar_cinza(str(falso))


def test_carregar_cinza_converte_cor_e_aceita_acentos(tmp_path):
    pasta = tmp_path / "Área de Trabalho"   # acentos quebram o cv2.imread no Windows
    pasta.mkdir()
    vermelho = np.zeros((6, 6, 3), dtype=np.uint8)
    vermelho[:, :, 2] = 255                 # BGR: canal 2 é o vermelho
    _, dados = cv2.imencode(".png", vermelho)
    dados.tofile(str(pasta / "vermelho.png"))

    cinza = carregar_cinza(str(pasta / "vermelho.png"))

    assert cinza.dtype == np.uint8
    assert cinza.ndim == 2
    assert cinza[0, 0] == 76                # 0,299 x 255 = 76 (luminância)


# -------------------------------------------------------------- mapa_direcao

def test_mapa_direcao_cor_segue_o_angulo_e_brilho_segue_a_magnitude():
    # Direita (0°), baixo (90°), esquerda (180°), cima (270°) e um pixel sem borda.
    gx = np.array([[1.0, 0.0, -1.0, 0.0, 0.0]])
    gy = np.array([[0.0, 1.0, 0.0, -1.0, 0.0]])

    mapa = mapa_direcao(gx, gy)
    assert mapa.shape == (1, 5, 3)
    assert mapa.dtype == np.uint8

    hsv = cv2.cvtColor(mapa, cv2.COLOR_BGR2HSV)
    matiz = hsv[0, :4, 0].astype(int)
    # No OpenCV a matiz vale ângulo / 2: 0°, 90°, 180°, 270° -> 0, 45, 90, 135
    assert np.all(np.abs(matiz - [0, 45, 90, 135]) <= 1)
    assert np.all(hsv[0, :4, 2] == 255)     # bordas fortes ficam brilhantes
    assert hsv[0, 4, 2] == 0                # sem gradiente fica preto


def test_mapa_direcao_formatos_diferentes_da_erro():
    with pytest.raises(ValueError):
        mapa_direcao(np.zeros((3, 3)), np.zeros((3, 4)))


# ------------------------------------------------------------- mostrar_grade

def test_mostrar_grade_salva_a_figura(pasta_saida):
    img = imagem_com_degrau()
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    colorida = mapa_direcao(gx, np.zeros_like(gx))

    caminho = mostrar_grade([img, gx, colorida], ["original", "gx", "cor"],
                            colunas=2, nome_saida="grade.png")

    assert Path(caminho).is_file()


def test_mostrar_grade_quantidade_de_titulos_errada_da_erro():
    with pytest.raises(ValueError, match="títulos"):
        mostrar_grade([imagem_com_degrau()], ["um", "dois"])


# ---------------------------------------------------- figuras 05 e 09 (integração)

def test_figuras_05_e_09_sao_geradas(pasta_saida):
    # Só roda quando os módulos do Fernando e do Igor já estão no repositório.
    pytest.importorskip("src.operadores")
    pytest.importorskip("src.scharr_parametros")

    img = np.zeros((60, 60), dtype=np.uint8)
    cv2.circle(img, (30, 30), 18, 255, -1)

    assert Path(utils.figura_comparacao(img)).name == "05_comparacao_operadores.png"
    assert Path(utils.figura_direcao(img)).name == "09_direcao.png"
    assert (pasta_saida / "05_comparacao_operadores.png").is_file()
    assert (pasta_saida / "09_direcao.png").is_file()
