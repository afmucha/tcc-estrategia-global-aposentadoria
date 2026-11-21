#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baixa das fontes oficiais as séries públicas usadas por simulacao_tcc.py e grava-as em dados/.

Séries: IPCA mensal (SGS 433), PTAX diária (SGS 1) e taxa DI acumulada no mês (SGS 4391), do Banco Central do
Brasil; CPI-U mensal (série CUUR0000SA0) do U.S. Bureau of Labor Statistics; e Ibovespa mensal da B3 (último
pregão de cada mês, ajustado pelas divisões do índice por 10 em 10/02/1994 e 03/03/1997).
A tábua de mortalidade do IBGE (2023) e o CPI-U já acompanham o repositório (o script tenta atualizar o CPI-U). A série do MSCI não é obtida por este script
(ver README.md). Uso: python3 baixar_dados_publicos.py   (requer acesso à internet; só a biblioteca padrão)
"""
import base64, csv, json, os, time, urllib.request
from datetime import date

FIM = 2025                                     # último ano necessário à simulação (janelas até jun./2025)
os.makedirs('dados', exist_ok=True)
UA = {'User-Agent': 'Mozilla/5.0 (reproducao academica)'}

def get(url):
    for tentativa in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
        except Exception as e:
            if tentativa == 3: raise
            time.sleep(3 * (tentativa + 1))

def sgs(codigo, ini, fim, passo_anos=10):
    """Série do SGS em janelas de até dez anos (limite da API para séries diárias)."""
    dados, a = [], ini.year
    while a <= fim.year:
        d0 = max(ini, date(a, 1, 1)); d1 = min(fim, date(min(a + passo_anos - 1, fim.year), 12, 31))
        url = (f'https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=json'
               f'&dataInicial={d0:%d/%m/%Y}&dataFinal={d1:%d/%m/%Y}')
        dados += json.loads(get(url)); a += passo_anos
    return dados

print('IPCA (SGS 433)...')
json.dump(sgs(433, date(1994, 1, 1), date(FIM, 12, 31)), open('dados/ipca_mensal.json', 'w'))
print('PTAX (SGS 1)...')
json.dump(sgs(1, date(1994, 1, 3), date(FIM, 12, 31)), open('dados/usdbrl_diario.json', 'w'))
print('CDI (SGS 4391)...')
with open('dados/cdi_mensal.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['data', 'valor'])
    for r in sgs(4391, date(1994, 7, 1), date(FIM, 12, 31)):
        d, m, a = r['data'].split('/'); w.writerow([f'{a}-{m}-{d}', r['valor']])

print('CPI-U (BLS)...')
try:
    cpi = {}
    for a0 in range(1969, FIM + 1, 10):
        url = f'https://api.bls.gov/publicAPI/v1/timeseries/data/CUUR0000SA0?startyear={a0}&endyear={min(a0 + 9, FIM)}'
        for it in json.loads(get(url))['Results']['series'][0]['data']:
            if it['period'].startswith('M') and it['period'] != 'M13' and it['value'] not in ('-', ''):
                cpi[f"{it['year']}-{it['period'][1:]}"] = it['value']
        time.sleep(1)
    with open('dados/cpi_eua_mensal.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['mes', 'indice'])
        for k in sorted(cpi): w.writerow([k, cpi[k]])
except Exception as e:                                     # a API do BLS recusa ou limita acessos automatizados
    print(f'  BLS indisponível ({type(e).__name__}); mantido o arquivo dados/cpi_eua_mensal.csv que acompanha o repositório.')

print('Ibovespa (B3)...')
num = lambda s: float(s.replace('.', '').replace(',', '.'))
linhas = []
for ano in range(1994, FIM + 1):
    p = base64.b64encode(json.dumps({'index': 'IBOV', 'language': 'pt-br', 'year': str(ano)}, separators=(',', ':')).encode()).decode()
    txt = base64.b64decode(get(f'https://sistemaswebb3-listados.b3.com.br/indexStatisticsProxy/IndexCall/GetDownloadPortfolioDay/{p}')
                           .strip().strip(b'"')).decode('latin-1')
    dias = [l.split(';') for l in txt.splitlines() if l[:1].isdigit()]
    for m in range(1, 13):
        valores = [(int(l[0]), l[m]) for l in dias if len(l) > m and l[m].strip()]
        if not valores: continue
        dia, v = max(valores)                                   # último pregão do mês
        d = date(ano, m, dia); pts = num(v)
        if d < date(1994, 2, 10): pts /= 10                     # divisão por 10 de 10/02/1994
        if d < date(1997, 3, 3): pts /= 10                      # divisão por 10 de 03/03/1997
        if d >= date(1994, 6, 1): linhas.append([f'{ano}-{m:02d}', d.isoformat(), str(round(pts, 2))])
    time.sleep(0.5)
with open('dados/ibovespa_mensal.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['mes', 'data_ultimo_pregao', 'pontos_ajustados']); w.writerows(linhas)
print('Concluído. Falta apenas dados/msci_world_net_mensal.csv (ver README.md).')
