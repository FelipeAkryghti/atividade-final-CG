# Detectores de bordas por gradiente — folha de estudo

**Fernando Barros Greca | Grupo 05 | Tema 03**

## Ideia central

Uma imagem em tons de cinza é uma função discreta `f(x,y)`. O gradiente
`∇f = (Gx, Gy)` estima como o brilho muda nas direções horizontal e vertical.
Ele aponta para o crescimento mais rápido da intensidade; a borda fica
perpendicular a essa direção.

## Diferenças finitas

Para passo de um pixel, a diferença central estima `∂f/∂x` por
`[f(x+1,y) − f(x−1,y)] / 2`. A estimativa em y é análoga. Os kernels 3×3
combinam essas diferenças em uma vizinhança.

## Kernels oficiais

| Operador | Gx | Gy |
|---|---|---|
| Prewitt | `[-1 0 1; -1 0 1; -1 0 1]` | `[-1 -1 -1; 0 0 0; 1 1 1]` |
| Sobel | `[-1 0 1; -2 0 2; -1 0 1]` | `[-1 -2 -1; 0 0 0; 1 2 1]` |

Sobel pesa mais a linha central e suaviza na direção perpendicular à derivada.
Seu `Gx` é o produto externo `[1,2,1]ᵀ · [-1,0,1]`; `Gy = Gxᵀ`.

## Correlação e convolução

`cv2.filter2D` calcula correlação: multiplica os pixels pelos pesos na
orientação escrita e soma. A convolução matemática gira o kernel 180° antes.
Nos kernels de derivada deste trabalho, o giro inverte os sinais e, portanto,
troca o sinal da resposta. Nas bordas, `BORDER_REFLECT_101` equivale a
`np.pad(..., mode="reflect")`.

## Saída e exemplo para explicar

`G = √(Gx² + Gy²)` mede a força da mudança; `θ = arctan2(Gy, Gx)` dá sua
direção em radianos. Em uma janela `[[0,0,255],[0,0,255],[0,0,255]]`, com
Sobel: `Gx = 255 + 510 + 255 = 1020`; `Gy = −255 + 255 = 0`; magnitude `1020`;
ângulo `0 rad = 0°`. A intensidade cresce para a direita.

## Código que devo saber explicar

`KERNELS` guarda os pesos em `float64`. `convolucao_manual` percorre a janela
com dois laços, multiplica pixel a pixel e soma, usando reflexão nas bordas.
`aplicar_prewitt` usa `cv2.filter2D`; `aplicar_sobel` usa `cv2.Sobel`.
`magnitude` usa `np.hypot`; `direcao` usa `np.arctan2`. `CV_64F` preserva
respostas negativas; em `uint8`, elas seriam cortadas para zero.
