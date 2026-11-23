# Explicação do código do Victor (linguagem simples)

## src/__init__.py
Só avisa ao Python: "a pasta `src` é um pacote". Sem ele, `from src.utils import ...` pode falhar.

## gerar_figuras.py
1. `matplotlib.use("Agg")`: desliga janelas; o script só grava arquivos.
2. `carregar_cinza` lê `imagens/entrada/teste.png` em tons de cinza (uint8).
3. `salvar(img, "01_original.png")` grava a figura 01 (a minha).
4. Chama as funções dos colegas: `figuras_sobel` (02 a 04, Fernando), `figura_comparacao` (05, Felipe), `figura_ksize`, `figura_blur_ruido`, `figura_limiar` (06 a 08, Igor) e `figura_direcao` (09, Felipe).
5. Imprime o caminho de cada arquivo. Aceita outra imagem como argumento: `python gerar_figuras.py outra.png`.

## notebooks/deteccao_bordas.ipynb
Célula a célula: ambiente (no Colab clona o repositório e instala `requirements.txt`) → imports → original → convolução manual × OpenCV (diferença 0) → Gx e Gy → magnitude e direção → três operadores → ksize → ruído e blur (`sigma`, `blur` alteráveis) → limiar (`limiar` alterável) → conclusão.

## app/streamlit_app.py
- Barra lateral: imagem (upload), operador, `ksize` (só Sobel), ruído σ, suavização Gaussiana e limiar.
- Pipeline: ruído → blur → gradientes (Gx, Gy em float64) → magnitude → limiar.
- `normalizar` converte para 0..255 só para mostrar; `mapa_direcao` devolve BGR, por isso convertemos para RGB antes de exibir.
- Usa apenas funções de `src/`.

## Imagem de teste
Sintética (retângulo, círculo, xadrez, triângulo, linhas diagonais e texto sobre fundo em degradê): tem bordas em várias direções e pode ser trocada por qualquer foto.
