# Trabalho 2 - Filtros Convolucionais e Transformada de Fourier

## Parte I - Filtros Convolucionais

Implementar os seguintes filtros convolucionais:

- Shift
- Caixa / Média
- Gaussiano
- Laplace
- Sobel

E os seguintes processos que envolvem filtros convolucionais:

- Aumento de Nitidez com Laplace
- Aumento de Nitidez com Máscara de Des-nitidez (Unsharp Mask)
- **Filtro criativo:** crie um filtro ou processo que envolva convolução e que não esteja na lista acima

### Roteiro de análise para cada filtro

Para cada filtro, responder as perguntas (P) e fazer as demonstrações (D):

| | |
|---|---|
| **P** | Qual é o propósito deste filtro? |
| **P** | Compare com outros filtros de impacto visual semelhante. Quais são as diferenças? |
| **PD** | Qual o impacto de diferentes formas de padding? Você usaria padding? Qual método e por quê? Mostre os resultados. |
| **PD** | Como este filtro se comporta no domínio das frequências? |
| **P** | Quais coeficientes são suprimidos? Quais são ressaltados? |
| **PD** | Este filtro aparece no seu dia a dia? Tem exemplos? |

> Para cada demonstração, escolha uma imagem que lhe agrade.

---

## Parte II - Transformada de Fourier Manual

Implementar manualmente a Transformada de Fourier e a Transformada Inversa.

Durante a Transformada Inversa, visualizar a reconstrução parcial da imagem começando pelas baixas frequências até as mais altas.

**Incluir no relatório:**
- 2 resultados parciais da IDFT com descrição do que chamou atenção

**Fazer para duas imagens:**
- Uma com coeficientes altos em **altas frequências** (ex.: fogos de artifício, confetti)
- Uma com coeficientes altos em **baixas frequências** (ex.: céus, mares, paisagens)

---

## Submissão

- Busque imagens interessantes para os exemplos (fotos próprias ou de fontes livres como [Unsplash](https://unsplash.com))
- Enviar código-fonte e relatório para o edisciplinas
- **Comentar o código**
- Organizar o código em funções com nomes descritivos (uma função por funcionalidade)

> Uma penalidade será aplicada se a professora não conseguir localizar as implementações.
