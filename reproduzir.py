#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Confere os dados, executa a simulação e compara os resultados com os publicados no TCC.

Etapas: (1) verifica se todos os arquivos de dados estão em dados/; (2) confere a série do MSCI obtida pelo leitor,
recalculando os 55 retornos anuais de fevereiro a fevereiro e comparando-os com o Apêndice C do trabalho;
(3) executa simulacao_tcc.py; (4) compara saidas/resultados.json com referencia/resultados_tcc.json.
Uso: python3 reproduzir.py
"""
import csv, json, os, subprocess, sys

NECESSARIOS = {
    'dados/ipca_mensal.json': 'público (Banco Central): execute baixar_dados_publicos.py',
    'dados/usdbrl_diario.json': 'público (Banco Central): execute baixar_dados_publicos.py',
    'dados/cdi_mensal.csv': 'público (Banco Central): execute baixar_dados_publicos.py',
    'dados/cpi_eua_mensal.csv': 'público (BLS): acompanha o repositório',
    'dados/tabua_mortalidade_ibge_2023.csv': 'público (IBGE): acompanha o repositório',
    'dados/ibovespa_mensal.csv': 'público (B3): execute baixar_dados_publicos.py',
    'dados/msci_world_net_mensal.csv': 'proprietário (MSCI): obtenha-o junto ao MSCI ou à Curvo (ver README.md)',
}
faltam = {f: m for f, m in NECESSARIOS.items() if not os.path.exists(f)}
if faltam:
    print('Arquivos de dados ausentes:')
    for f, m in faltam.items(): print(f'  - {f}: {m}')
    sys.exit(1)

# ---- (2) conferência da série do MSCI com os retornos anuais publicados no Apêndice C ----
niv = {r['mes']: float(r['nivel_usd']) for r in csv.DictReader(open('dados/msci_world_net_mensal.csv'))}
AJ = 11936.580776 / 12057.735537          # 27/02/2025 ÷ 28/02/2025 (nível diário oficial), como em simulacao_tcc.py
try:
    fev = [niv[f'{a}-02'] for a in range(1970, 2026)]; fev[-1] *= AJ
    ret = [100 * (fev[i + 1] / fev[i] - 1) for i in range(55)]
    nom94 = (niv['2025-06'] / niv['1994-06']) ** (1 / 31) - 1
except KeyError as e:
    sys.exit(f'A série do MSCI precisa cobrir ao menos dez./1969 a jun./2025 (falta o mês {e}).')
pub = [float(r['msci_usd']) for r in csv.DictReader(open('referencia/apendice_c_series.csv')) if r['ano'][0].isdigit()]
dif = max(abs(a - b) for a, b in zip(ret, pub))
print(f'Série do MSCI: maior diferença frente ao Apêndice C = {dif:.3f} p.p. | retorno nominal jul./1994–jun./2025 = {nom94:.2%} (TCC: 8,10%)')
if dif > 0.006:
    print('  ATENÇÃO: a série obtida difere da utilizada no trabalho; os resultados não devem coincidir exatamente.')

# ---- (3) execução da simulação ----
print('Executando simulacao_tcc.py (cerca de um minuto)...')
os.makedirs('saidas', exist_ok=True)
with open('saidas/log_simulacao.txt', 'w') as log:
    rc = subprocess.run([sys.executable, 'simulacao_tcc.py'], stdout=log, stderr=subprocess.STDOUT).returncode
if rc != 0:
    sys.exit('A simulação terminou com erro; veja saidas/log_simulacao.txt.')

# ---- (4) comparação com os resultados do TCC ----
def plano(d, p=''):
    if isinstance(d, dict): return {k2: v for k, x in d.items() for k2, v in plano(x, f'{p}/{k}').items()}
    if isinstance(d, list): return {k2: v for i, x in enumerate(d) for k2, v in plano(x, f'{p}[{i}]').items()}
    return {p: d}
novo = plano(json.load(open('saidas/resultados.json'))); ref = plano(json.load(open('referencia/resultados_tcc.json')))
def igual(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)): return abs(a - b) <= 1e-9 * max(1.0, abs(b))
    return a == b
diferentes = [k for k in ref if k not in novo or not igual(novo[k], ref[k])]
print(f'Resultados: {len(ref) - len(diferentes)} de {len(ref)} valores idênticos aos publicados no TCC.')
for k in diferentes[:20]: print(f'  {k}: obtido {novo.get(k)} | TCC {ref[k]}')
print('Saída completa da simulação em saidas/log_simulacao.txt.')
