"""Aplicativo Streamlit: detecção de bordas por gradiente (autor: Victor).

Executar da raiz do repositório:  streamlit run app/streamlit_app.py
Usa apenas funções de src/ (utils: Felipe, operadores: Fernando, scharr: Igor).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Garante que "from src..." funcione ao rodar de qualquer pasta
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.operadores import aplicar_prewitt, aplicar_sobel, magnitude
from src.scharr_parametros import adicionar_ruido, aplicar_scharr, limiarizar
from src.utils import mapa_direcao, normalizar

st.set_page_config(page_title="Detectores de bordas", layout="wide")
st.title("Detectores de bordas por gradiente")
st.caption("Grupo 05 | Tema 03 | Prewitt, Sobel e Scharr")

# ---------------- Barra lateral: parâmetros ----------------
with st.sidebar:
    arquivo = st.file_uploader("Imagem", type=["png", "jpg", "jpeg", "bmp"])
    operador = st.selectbox("Operador", ["Sobel", "Prewitt", "Scharr"])
    # ksize só vale para o Sobel (cv2.Sobel); Prewitt e Scharr são 3x3 fixos
    ksize = st.select_slider("ksize (só Sobel)", options=[1, 3, 5, 7], value=3,
                             disabled=operador != "Sobel")
    sigma = st.slider("Ruído gaussiano (σ)", 0, 80, 0)
    blur = st.select_slider("Suavização Gaussiana (tamanho)", options=[0, 3, 5, 7, 9], value=0)
    limiar = st.slider("Limiar (0..255)", 0, 255, 80)

# ---------------- Entrada ----------------
if arquivo is not None:
    dados = np.frombuffer(arquivo.read(), np.uint8)
    img = cv2.imdecode(dados, cv2.IMREAD_GRAYSCALE)
else:
    img = cv2.imread(str(RAIZ / "imagens" / "entrada" / "teste.png"), cv2.IMREAD_GRAYSCALE)
if img is None:
    st.error("Não foi possível ler a imagem.")
    st.stop()

# ---------------- Pipeline: ruído -> blur -> gradientes -> magnitude -> limiar ----------------
trabalho = adicionar_ruido(img, sigma=sigma) if sigma > 0 else img.copy()
if blur > 0:
    trabalho = cv2.GaussianBlur(trabalho, (blur, blur), 0)

if operador == "Sobel":
    gx, gy = aplicar_sobel(trabalho, ksize=ksize)
elif operador == "Prewitt":
    gx, gy = aplicar_prewitt(trabalho)
else:
    gx, gy = aplicar_scharr(trabalho)

mag = magnitude(gx, gy)  # G = sqrt(Gx² + Gy²), float64
# limiarizar não reescala: normalizamos a magnitude para 0..255 antes de comparar com o limiar
binaria = limiarizar(normalizar(mag), limiar)


# ---------------- Exibição (normalizar só serve para mostrar) ----------------
def mostrar(coluna, imagem, titulo):
    """Mostra uma imagem 0..255 numa coluna, com legenda."""
    coluna.image(imagem, caption=titulo, clamp=True, use_container_width=True)


linha1 = st.columns(3)
mostrar(linha1[0], img, "Original")
mostrar(linha1[1], trabalho, f"Entrada do operador (σ={sigma}, blur={blur})")
mostrar(linha1[2], normalizar(mag), f"Magnitude ({operador})")

linha2 = st.columns(3)
mostrar(linha2[0], normalizar(gx), "Gx (bordas verticais)")
mostrar(linha2[1], normalizar(gy), "Gy (bordas horizontais)")
mostrar(linha2[2], binaria, f"Limiar = {limiar}")

st.subheader("Direção do gradiente")
# mapa_direcao devolve BGR; o Streamlit espera RGB
st.image(cv2.cvtColor(mapa_direcao(gx, gy), cv2.COLOR_BGR2RGB),
         caption="Matiz = direção θ, brilho = magnitude", width=480)
