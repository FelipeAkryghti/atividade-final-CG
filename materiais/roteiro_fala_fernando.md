# Roteiro de fala — slides 5 a 10 (aprox. 4 minutos)

## Slide 5 — Gradiente de imagem

Considere a imagem em tons de cinza como uma função f de x e y: em cada
posição existe um valor de intensidade. O gradiente reúne duas derivadas
parciais. Gx mede quanto o brilho muda quando caminhamos na horizontal; Gy
mede a mudança quando caminhamos na vertical. A direção do gradiente aponta
para onde a intensidade cresce mais rapidamente. Como essa mudança costuma
acontecer atravessando o contorno de um objeto, o módulo do gradiente destaca
bordas. Nos próximos slides, vamos estimar essas derivadas em pixels usando
diferenças finitas e kernels pequenos.

## Slide 6 — Diferenças finitas

A derivada contínua vira uma estimativa usando pixels vizinhos. Para uma
diferença central, com passo de um pixel, comparamos o brilho à direita e à
esquerda e dividimos por dois. No exemplo, o pixel central vale 50 e seus
vizinhos valem 50 e 180; a taxa estimada é 65 níveis de intensidade por pixel.
Fazemos o mesmo na vertical para obter Gy. Nas bordas da imagem não há
vizinhos dos dois lados, então o operador usa uma regra de extensão da imagem.

## Slide 7 — Correlação com kernel 3×3

Um kernel percorre a imagem e encontra uma resposta para cada posição. Na
posição atual, multiplicamos cada um dos nove pixels pelo peso que ocupa a
mesma posição no kernel e somamos os produtos. Essa operação é correlação,
como cv2.filter2D: os pesos não são girados. Convolução matemática gira o
kernel em 180 graus; para kernels de derivada ímpares como os que usamos, isso
inverte o sinal da resposta. O preenchimento REFLECT_101 espelha os pixels na
borda sem repetir o pixel extremo.

## Slide 8 — Prewitt

Prewitt usa pesos iguais ao longo da direção perpendicular à derivada. Os
sinais negativos e positivos medem a diferença entre lados opostos; o centro
e a soma total iguais a zero fazem áreas uniformes produzirem resposta nula.
Gx responde com força a mudanças horizontais de intensidade, que aparecem em
contornos verticais. Gy faz o papel correspondente para contornos horizontais.
A simplicidade ajuda a explicar o operador, embora ele faça menos suavização
que o Sobel.

## Slide 9 — Sobel e sua forma separável

Sobel mantém a estrutura de diferença entre esquerda e direita do Prewitt,
mas aumenta o peso da linha central de um para dois. Isso aplica uma
suavização perpendicular à derivada e reduz a influência de pequenas
oscilações. O kernel horizontal pode ser construído como produto externo do
vetor de suavização [1, 2, 1] pela diferença [-1, 0, 1]. Gy é a transposta do
Gx. A forma separável também permite implementar o cálculo em duas passagens
unidimensionais, embora aqui usemos diretamente cv2.Sobel.

## Slide 10 — Magnitude, direção e exemplo numérico

Vamos calcular no papel uma vizinhança 3×3 cuja última coluna é mais clara:
[0, 0, 255] em cada linha. Com o kernel Sobel, os produtos de Gx na coluna
direita somam 255 + 510 + 255, então Gx vale 1020. Os produtos de Gy nessa
coluna são −255 e +255 e se cancelam, então Gy é zero. A magnitude é 1020 e o
ângulo é zero radianos: o gradiente aponta para a direita, onde o brilho
aumenta. O código usa CV_64F para conservar esse sinal e escala.
