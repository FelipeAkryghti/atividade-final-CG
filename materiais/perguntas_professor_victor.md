# 5 perguntas prováveis do professor (código e integração)

1. **Por que `cv2.CV_64F` nos gradientes?**
   uint8 só guarda 0..255; transições claro→escuro dão valores negativos e seriam cortadas em 0, apagando metade das bordas. Em float64 guardamos o sinal e só normalizamos para exibir.

2. **Como integraram o trabalho de 4 pessoas sem quebrar o código?**
   Contrato de assinaturas combinado antes, um módulo por pessoa, uma branch e um Pull Request por módulo, revisão e merge pelo Victor, `pytest` por módulo e `gerar_figuras.py` como teste de integração, pois chama todos os módulos.

3. **O que faz `normalizar` e por que é necessária?**
   Faz min-max para 0..255 (`cv2.normalize`, NORM_MINMAX) e converte para uint8. Gradientes passam de 255 e têm negativos; a tela só mostra 0..255. É só para visualização; os cálculos seguem em float64.

4. **Como a convolução manual se relaciona com o OpenCV?**
   Desliza o kernel 3×3 e soma produtos (na prática correlação, a mesma convenção do `cv2.filter2D`, sem espelhar o kernel). A diferença contra o OpenCV é 0, o que valida a implementação.

5. **Por que suavizar antes e como escolher o limiar?**
   A derivada amplifica ruído; a Gaussiana reduz bordas falsas, mas engrossa as reais. O limiar é um compromisso: baixo gera ruído, alto perde bordas fracas; ajusta-se olhando o resultado (sliders do Streamlit).

Extras curtos: *Por que Gy é a transposta de Gx?* É a mesma derivada em outro eixo. *Por que Scharr?* Mais fiel em diagonais. *O que o `__init__.py` faz?* Marca `src` como pacote para os imports a partir da raiz.
