# Perguntas prováveis do professor — Fernando

1. **O que Gx e Gy representam?**
   Gx mede variações horizontais de intensidade; Gy mede variações verticais.

2. **Por que a magnitude usa raiz da soma dos quadrados?**
   Porque combina as componentes ortogonais do gradiente e mede sua força sem
   depender do sinal de cada componente: `√(Gx² + Gy²)`.

3. **Qual é a diferença entre convolução e correlação?**
   Na convolução, giramos o kernel 180°; na correlação, usamos os pesos como
   estão. `cv2.filter2D` calcula correlação.

4. **Por que usar `CV_64F`?**
   Para manter valores negativos e amplitudes maiores que 255. Em `uint8`, os
   negativos seriam truncados e parte das bordas desapareceria.

5. **O que a forma separável do Sobel significa?**
   Que o kernel 3×3 pode ser obtido por dois vetores 1D: suavização
   `[1,2,1]ᵀ` e derivada `[-1,0,1]`, aplicados em sequência.
