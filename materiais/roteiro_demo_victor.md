# Roteiro da demonstração ao vivo (5 min, Victor)

Antes: notebook aberto no Colab/Jupyter com **todas as células já executadas uma vez**, Streamlit rodando em outra aba
(`streamlit run app/streamlit_app.py`), figuras de `imagens/saida/` abertas como reserva.

| Tempo | Célula | O que falar |
|---|---|---|
| 0:00–0:30 | Preparação + imports | "O código é do grupo: Felipe fez `utils`, Fernando `operadores`, Igor `scharr_parametros`; eu integrei. Tudo importa a partir da raiz." |
| 0:30–1:00 | 1. Original | "Tons de cinza, uint8, com bordas verticais, horizontais, diagonais e curvas." Mostrar shape e dtype. |
| 1:00–1:30 | 2. Convolução manual × OpenCV | "Nossa convolução em NumPy puro dá diferença 0 contra o OpenCV: é soma de produtos da janela 3×3." |
| 1:30–2:15 | 3. Gx e Gy | "Gx responde a bordas verticais, Gy a horizontais. Usamos CV_64F: em uint8 os negativos seriam cortados e perderíamos metade das bordas." Apontar o min negativo. |
| 2:15–2:50 | 4. Magnitude e direção | "G = √(Gx²+Gy²) junta os dois; θ = arctan2(Gy,Gx) dá a orientação, colorida no mapa." |
| 2:50–3:20 | 5. Prewitt × Sobel × Scharr | "Prewitt pesos iguais, Sobel peso 2 no centro, Scharr 10 e 3: mais preciso nas diagonais." |
| 3:20–4:30 | 7. Ruído/blur e 8. Limiar (altere `sigma` 25→50 e `limiar` 80→40, reexecute) | "Mais ruído, mais bordas falsas. Suavizar antes reduz, mas engrossa as bordas. Limiar baixo: tudo vira borda; alto: perde as fracas." |
| 4:30–5:00 | Streamlit (opcional) e passagem | "Mesmo pipeline interativo: upload, operador, sliders. Passo para o Felipe." |

Se estiver atrasado: pule a convolução manual e o Streamlit (a seção 6, ksize, é opcional aqui; Igor cobre nos slides 18 a 20).

## Plano B
- **Colab não abre / clone falha:** rodar localmente (`jupyter notebook notebooks/deteccao_bordas.ipynb`) com o ambiente já instalado antes da aula.
- **Sem internet:** tudo roda local; só o clone no Colab depende de rede. Ter o repositório clonado e `pip install -r requirements.txt` feito.
- **Notebook trava:** mostrar as 9 figuras de `imagens/saida/` e narrar o mesmo roteiro; `python gerar_figuras.py` regenera tudo em segundos.
- **Meu notebook falha:** outro integrante com o repositório clonado, ou usar os slides 13 a 17.
- Levar o repositório em pendrive/ZIP e o PDF dos slides.
