# Estratégia passiva global e regra de distribuição dinâmica como complemento à aposentadoria

Código e dados públicos da simulação do Trabalho de Conclusão de Curso de **André Felipe Mucha**
(Administração, Universidade Federal da Fronteira Sul – UFFS, 2026).

O script `simulacao_tcc.py` é o mesmo transcrito na íntegra no Apêndice A do trabalho. Ele produz todos os resultados
numéricos da simulação reportados no Capítulo 4: acumulação, Monte Carlo, janelas históricas, regras de retirada,
probabilidades de ruína, testes de robustez, comparações com o INSS, o Tesouro RendA+, a poupança, o BDR BURT39, o
Ibovespa e o CDI, e ruína ao longo da vida.

## Conteúdo

| Arquivo | Descrição |
|---|---|
| `simulacao_tcc.py` | Script canônico da simulação (Apêndice A) |
| `reproduzir.py` | Confere os dados, executa a simulação e compara os resultados com os publicados no TCC |
| `baixar_dados_publicos.py` | Baixa das fontes oficiais as séries do Banco Central, do BLS e da B3 |
| `dados/` | Séries públicas: IPCA (SGS 433), PTAX (SGS 1) e CDI (SGS 4391) do Banco Central; CPI-U do BLS; tábua de mortalidade do IBGE (2023) |
| `referencia/resultados_tcc.json` | Resultados da simulação tal como publicados no trabalho |
| `referencia/apendice_c_series.csv` | Retornos anuais do MSCI World Net (Apêndice C), usados para conferir a série obtida pelo leitor |

## Como reproduzir

Ambiente utilizado no trabalho: **Python 3.14.3, NumPy 2.4.3 e pandas 3.0.1**, com sementes fixas (42 e 7).
Outras versões das bibliotecas podem gerar sequências aleatórias distintas e pequenas diferenças nos resultados de
Monte Carlo.

1. `pip install numpy==2.4.3 pandas==3.0.1`
2. `python3 baixar_dados_publicos.py`: gera `dados/ibovespa_mensal.csv` e atualiza as séries do Banco Central.
3. Obtenha a série do MSCI (ver seção seguinte) e salve-a como `dados/msci_world_net_mensal.csv`.
4. `python3 reproduzir.py`

O `reproduzir.py` informa quantos dos 443 valores coincidem com os publicados no trabalho. Com os mesmos dados e as
mesmas versões, todos coincidem.

## Por que a série do MSCI não está neste repositório

A série mensal do índice **MSCI World Net Return**, base da simulação, é material proprietário da MSCI Inc.
Os termos de uso do MSCI (revisão de 15 de julho de 2026) reconhecem os dados de seus índices como propriedade
intelectual do grupo MSCI. Eles vedam redistribuir, revender, transferir ou disponibilizar esse material a qualquer
outra pessoa ou entidade sem licença (<https://www.msci.com/legal/terms-of-use>). Por isso, o arquivo usado no trabalho
não é publicado aqui nem enviado a terceiros, nem mesmo mediante solicitação.

Para reproduzir os resultados, o leitor deve obter a série por conta própria:

- **Fontes:** junto ao próprio MSCI (dados licenciados) ou consultando a plataforma Curvo
  (<https://curvo.eu/backtest/pt/indice/msci-world?currency=usd>), por meio da qual a série do trabalho foi obtida e
  conferida com os dados oficiais do MSCI.
- **Formato:** arquivo CSV com as colunas `mes` (AAAA-MM) e `nivel_usd`, que é o nível de fechamento mensal do índice
  em dólares, em qualquer base.
- **Período:** ao menos de dezembro de 1969 a junho de 2025.

O `reproduzir.py` confere automaticamente a série obtida: recalcula os 55 retornos anuais de fevereiro a fevereiro e
os compara com os publicados no Apêndice C do trabalho (arquivo `referencia/apendice_c_series.csv`).

O Ibovespa segue critério semelhante, por cautela. Os dados de índices da B3 também estão sujeitos à política de
licenciamento da bolsa, e por isso o arquivo não é redistribuído: `baixar_dados_publicos.py` gera-o a partir da página
pública de estatísticas históricas da B3.

## Fontes dos dados públicos

- Banco Central do Brasil, Sistema Gerenciador de Séries Temporais (SGS): IPCA (série 433), PTAX de venda (série 1) e
  taxa DI acumulada no mês (série 4391).
- U.S. Bureau of Labor Statistics: Consumer Price Index for All Urban Consumers (CPI-U, série CUUR0000SA0).
- IBGE: Tábua completa de mortalidade para o Brasil, 2023 (ambos os sexos, homens e mulheres).
- B3: Índice Bovespa, estatísticas históricas (último pregão de cada mês, ajustado pelas divisões do índice por 10 em
  10/02/1994 e 03/03/1997).

## Licença e citação

O código está sob a licença MIT (arquivo `LICENSE`). Os dados pertencem às respectivas fontes e seguem os termos de uso
de cada uma. Para citar, veja `CITATION.cff`.
