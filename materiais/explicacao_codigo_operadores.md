# Como explicar `src/operadores.py`

Este guia acompanha o arquivo [src/operadores.py](../src/operadores.py) e o
exemplo [exemplo_numerico.py](../exemplo_numerico.py). As faixas citadas são
as linhas atuais do módulo.

## Linhas 1–11: preparação

- As primeiras linhas identificam o objetivo do módulo: estimar gradientes e
  manter suas respostas em `float64`, inclusive quando forem negativas.
- `cv2` fornece os filtros Sobel e Prewitt. `numpy` guarda matrizes e calcula
  a magnitude e o ângulo.

## Linhas 14–29: `KERNELS`

- O dicionário associa o nome de cada operador ao par `(Gx, Gy)`.
- Cada matriz é explicitamente criada como `np.float64`.
- Gy é a transposta de Gx: um kernel mede a variação horizontal, o outro a
  vertical. Incluí Scharr no dicionário porque ele faz parte do contrato do
  repositório, embora sua função de aplicação fique no módulo de Igor.

## Linhas 32–72: `convolucao_manual`

1. As linhas 53–54 convertem imagem e kernel para `float64`, para não perder
   precisão ou sinais durante as multiplicações.
2. As linhas 55–60 rejeitam imagem colorida, kernel inválido ou entradas vazias.
3. As linhas 62–64 calculam quantos pixels cercam o centro do kernel e
   estendem a imagem com `np.pad(mode="reflect")`. Isso reproduz
   `BORDER_REFLECT_101`: o pixel extremo não é repetido.
4. A linha 65 prepara uma matriz de saída zerada.
5. Os laços das linhas 66–67 visitam cada posição do kernel. Para cada peso,
   as linhas 68–71 multiplicam esse peso por toda a fatia correspondente da
   imagem ampliada e acumulam o resultado. A fatia vetorizada calcula muitos
   pixels de saída por vez; os laços explícitos percorrem as coordenadas do
   kernel.
6. A linha 72 devolve a resposta em `float64`.

OpenCV chama essa operação de correlação: aplica o kernel como foi escrito.
Convolução matemática gira o kernel 180 graus. Para os kernels de derivada
Prewitt, Sobel e Scharr, esse giro troca cada sinal pelo oposto; por isso a
convolução produz o negativo da correlação.

## Linhas 75–90: `aplicar_prewitt`

- Confere que a entrada é uma imagem 2D em tons de cinza.
- Busca os dois kernels Prewitt no dicionário.
- Aplica cada kernel com `cv2.filter2D` e saída `cv2.CV_64F`.
- Retorna `(gx, gy)` sem converter os gradientes para imagem de exibição.

## Linhas 93–115: `figuras_sobel`

- Importa `normalizar` e `salvar` do módulo de utilidades somente quando a
  função é chamada.
- Calcula Gx e Gy com o operador Sobel e combina as componentes pela função
  `magnitude`.
- Normaliza cada mapa para a faixa visível 0–255 e grava os três nomes
  combinados: `02_sobel_gx.png`, `03_sobel_gy.png` e
  `04_sobel_magnitude.png`.
- A normalização é apenas para exibição; os arrays do gradiente continuam em
  `float64` durante os cálculos.

## Linhas 118–135: `aplicar_sobel`

- Valida a imagem e o tamanho solicitado para o kernel.
- `dx=1, dy=0` pede a derivada horizontal; `dx=0, dy=1` pede a vertical.
- `cv2.CV_64F` mantém valores negativos e amplitudes acima de 255.
- Retorna as duas componentes para uso nas funções seguintes.

## Linhas 138–169: magnitude e direção

- `magnitude` verifica que Gx e Gy têm o mesmo formato e usa
  `np.hypot(gx, gy)`, equivalente a `sqrt(gx**2 + gy**2)`.
- `direcao` faz a mesma validação e usa `np.arctan2(gy, gx)`. O resultado
  está em radianos, no intervalo de `-π` a `π`.

## Como explicar `exemplo_numerico.py`

O script cria uma janela 3×3 com duas colunas escuras e uma clara. Multiplica
essa janela pelos kernels Sobel, imprime os produtos e soma cada matriz. O
resultado é `Gx=1020`, `Gy=0`, magnitude `1020` e ângulo `0°`. Assim, a direção
do gradiente aponta para a direita, do escuro para o claro.
