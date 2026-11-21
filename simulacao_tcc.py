#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulação canônica do TCC — André Felipe Mucha
Estratégia passiva global (SPDR MSCI World UCITS ETF, SWRD) + regra de
distribuição dinâmica (Vanguard Dynamic Spending).

Parâmetros conforme a metodologia (Cap. 3):
  - aporte de R$ 300/mês, ao fim de cada mês, reajustado a cada janeiro pelo IPCA acumulado de janeiro a
    dezembro do ano anterior (inflação suposta constante, igual ao IPCA médio do período do real);
    30 anos de acumulação e 30 de usufruto
  - custo de conversão cambial 2,10% por aporte (spread médio 1,00% + IOF 1,10%) e 1,38% na volta
  - TER 0,12% a.a.; piso de gasto -2,5% e teto +5% (Jaconetti et al., 2016)
  - INSS 2025: médio R$ 1.634; piso R$ 1.518
Todos os valores em R$ REAIS (poder de compra de hoje). Semente fixa (42).
Ações: MSCI World Net Return (USD), benchmark do SWRD, em base mensal desde 31/12/1969 (série
obtida via Curvo e conferida com os dados oficiais do MSCI, idêntica a partir de 1987).
Cada retorno nominal é deflacionado pela inflação dos EUA do mesmo período (CPI-U, BLS).
Janelas de referência: 27/02/1970 a 27/02/2025 (desde o lançamento do índice) e 01/07/1994 a
01/07/2025 (período do real, mesma janela do câmbio, do IPCA e da inflação norte-americana).
Nas simulações de Monte Carlo, a série real é deslocada uniformemente até que sua média
geométrica iguale os 5% reais do cenário base; as janelas históricas usam os retornos ocorridos.
As rendas da fase de usufruto são brutas de IR e do custo de saída.
"""
import json, os
import numpy as np, pandas as pd

# ---------------- 1. DADOS REAIS ----------------
msci_m = pd.read_csv('dados/msci_world_net_mensal.csv').set_index('mes')['nivel_usd']  # MSCI World Net (USD)
cpi_m = pd.read_csv('dados/cpi_eua_mensal.csv').set_index('mes')['indice']           # CPI-U mensal (BLS)
AJ_27FEV2025 = 11936.580776 / 12057.735537      # MSCI oficial (diário, NETR USD): 27/02/2025 ÷ 28/02/2025
# Anos de fevereiro a fevereiro, a partir do lançamento do índice (27/02/1970, último pregão de
# fevereiro de 1970) até 27/02/2025: 55 retornos anuais; nada anterior ao lançamento é utilizado.
anos_arr = np.arange(1970, 2025)                         # ano de início de cada período anual
fev = np.array([msci_m[f'{a}-02'] for a in range(1970, 2026)]); fev[-1] *= AJ_27FEV2025
cfev = np.array([cpi_m[f'{a}-02'] for a in range(1970, 2026)])
nom = fev[1:] / fev[:-1] - 1                             # retornos anuais líquidos (USD)
real_hist = (1 + nom) / (cfev[1:] / cfev[:-1]) - 1      # deflacionados pela inflação dos EUA do mesmo período

# ---------------- 2. PARÂMETROS ----------------
APORTE = 300.0; MESES = 360; ANOS_RET = 30
TER = 0.0012
C_APORTE = 0.0100 + 0.0110                             # 2,10% por aporte (spread médio 1,00% + IOF 1,10%)
C_SAIDA = 0.0100 + 0.0038                              # 1,38% na conversão de volta (spread 1,00% + IOF 0,38%)
IR = 0.15                                              # Lei nº 14.754/2023
ap_liq = APORTE * (1 - C_APORTE)
TETO, PISO = 0.05, -0.025                              # guardrails Vanguard
INSS_MEDIO, INSS_PISO, INSS_TETO = 1634.00, 1518.00, 8157.41
N, BLOCK = 10000, 5
RET_BASE = 0.05                                        # retorno real do cenário base (antes do TER)
R = {}                                                 # resultados salvos em saidas/resultados.json

def geo(x):
    """Média geométrica de uma série de retornos anuais."""
    return np.prod(1 + x) ** (1 / len(x)) - 1

def recentra(serie, alvo=RET_BASE):
    """Desloca a série uniformemente até que sua média geométrica iguale o alvo, preservando
    volatilidade, caudas e a ordem dos anos dentro de cada bloco (bissecção)."""
    lo, hi = -0.10, 0.10
    for _ in range(60):
        d = (lo + hi) / 2
        if geo(serie - d) > alvo: lo = d
        else: hi = d
    return serie - d, d

real_mc, D_MC = recentra(real_hist)                    # série usada no Monte Carlo
R['deflator'] = {'geo_real_hist': geo(real_hist), 'desloc_mc': D_MC, 'geo_real_mc': geo(real_mc)}
print(f"=== SÉRIE REAL 27/02/1970–27/02/2025: média geométrica {geo(real_hist):.2%}; deslocamento p/ o cenário base {D_MC:+.2%}")

def mensal(a):
    """Taxa mensal equivalente a uma taxa anual: (1 + i_a)^(1/12) - 1."""
    return (1 + a) ** (1/12) - 1

def liq_ter(a, ter=TER):
    """Retorno anual líquido da taxa de administração: (1 + a)(1 - TER) - 1."""
    return a - ter * (1 + a)

def blocos(serie, n, rng, bloco=BLOCK):
    """Reamostragem circular em blocos de anos consecutivos (Politis; Romano, 1992): a série é
    tratada como contínua, de modo que todos os anos têm a mesma probabilidade de sorteio."""
    out = []; L = len(serie)
    while len(out) < n:
        i = rng.integers(0, L)
        out.extend(serie[(i + np.arange(bloco)) % L])
    return np.array(out[:n])

def acumula(retornos_anuais, aporte=ap_liq, ter=TER):
    """Cada retorno anual vira taxa mensal equivalente, aplicada aos 12 meses do ano; o aporte entra ao
    fim de cada mês, com o valor real do mês k do ano (PERFIL, seção 2b): S_t = S_(t-1) * (1 + i_m) + P * PERFIL[k]."""
    s = 0.0
    for a in retornos_anuais:
        rm = mensal(liq_ter(a, ter))
        for k in range(12):
            s = s * (1 + rm) + aporte * PERFIL[k]
    return s

# ---------------- 2b. JANELAS DE REFERÊNCIA: AÇÕES, CÂMBIO E INFLAÇÃO ----------------
def janela_acoes(ini, fim, anos, ajuste=1.0):
    """Retorno anual nominal e real (deflacionado pelo CPI) do MSCI World Net entre dois fechamentos."""
    f = msci_m[fim] / msci_m[ini] * ajuste
    meses = list(msci_m.index); r = msci_m.values[1:] / msci_m.values[:-1] - 1
    vol = np.std(r[meses.index(ini):meses.index(fim)], ddof=1) * np.sqrt(12)
    return f ** (1 / anos) - 1, (f / (cpi_m[fim] / cpi_m[ini])) ** (1 / anos) - 1, vol
# (a) desde o lançamento: 27/02/1970 (último pregão de fev./1970) a 27/02/2025 — 55 anos
eq70_nom, eq70_real, eq70_vol = janela_acoes('1970-02', '2025-02', 55, AJ_27FEV2025)
# (b) período do real: 01/07/1994 a 01/07/2025 (fechamentos de 30/06) — 31 anos
eq94_nom, eq94_real, eq94_vol = janela_acoes('1994-06', '2025-06', 31)
# câmbio real mensal = PTAX (último dia útil do mês; 01/07/1994 = R$ 1,00) × CPI EUA ÷ IPCA
ptax = pd.DataFrame(json.load(open('dados/usdbrl_diario.json')))
ptax['data'] = pd.to_datetime(ptax['data'], dayfirst=True); ptax['valor'] = ptax['valor'].astype(float)
ptax = ptax[ptax.data >= '1994-07-01']
ptax_m = ptax.groupby(ptax.data.dt.strftime('%Y-%m'))['valor'].last()
ipca_mm = pd.DataFrame(json.load(open('dados/ipca_mensal.json')))
ipca_mm['mes'] = pd.to_datetime(ipca_mm['data'], dayfirst=True).dt.strftime('%Y-%m')
ipca_mm = ipca_mm.set_index('mes')['valor'].astype(float) / 100
meses94 = [f'{a}-{m:02d}' for a in range(1994, 2026) for m in range(1, 13) if '1994-07' <= f'{a}-{m:02d}' <= '2025-06']
nivel_ipca = np.cumprod(1 + ipca_mm[meses94].values)              # base: fim de jun./1994 = 1
rer = np.r_[1.00 * cpi_m['1994-06'], ptax_m[meses94].values * cpi_m[meses94].values / nivel_ipca]
t = np.arange(len(rer)) / 12
fx_tend = np.exp(np.polyfit(t, np.log(rer), 1)[0]) - 1          # tendência log-linear (MQO), 373 meses
# Incerteza da tendência: erro-padrão de Newey-West (autocorrelação dos resíduos) e teste ADF de raiz unitária
Xt = np.c_[np.ones(len(t)), t]; yt = np.log(rer); bt = np.linalg.lstsq(Xt, yt, rcond=None)[0]; et = yt - Xt @ bt
def ep_newey_west(L):
    XtX = np.linalg.inv(Xt.T @ Xt); S = (Xt * et[:, None]).T @ (Xt * et[:, None])
    for l in range(1, L + 1):
        G = (Xt[l:] * et[l:, None]).T @ (Xt[:-l] * et[:-l, None]); S += (1 - l / (L + 1)) * (G + G.T)
    return np.sqrt((XtX @ S @ XtX)[1, 1])
fx_ic = {str(L): [np.exp(bt[1] - 1.96 * ep_newey_west(L)) - 1, np.exp(bt[1] + 1.96 * ep_newey_west(L)) - 1] for L in (12, 36, 60, 120)}
dy, k_adf = np.diff(yt), 12                                     # ADF com constante, tendência e 12 defasagens
Zt = np.c_[np.ones(len(dy) - k_adf), t[k_adf + 1:], yt[k_adf:-1], np.column_stack([dy[k_adf - j - 1:len(dy) - j - 1] for j in range(k_adf)])]
ct = np.linalg.lstsq(Zt, dy[k_adf:], rcond=None)[0]; rt = dy[k_adf:] - Zt @ ct
adf = ct[2] / np.sqrt(rt @ rt / (len(rt) - Zt.shape[1]) * np.linalg.inv(Zt.T @ Zt)[2, 2])   # valor crítico 5% ≈ −3,42
ipca94 = nivel_ipca[-1] ** (1 / 31) - 1
cpi94 = (cpi_m['2025-06'] / cpi_m['1994-06']) ** (1 / 31) - 1
fx94_nom = (ptax_m['2025-06'] / 1.00) ** (1 / 31) - 1
OTIMISTA = (1 + eq94_real) * (1 + fx_tend) - 1                  # repetição do período do real
IPCA = ipca94                                                   # usada no reajuste dos aportes e no IR das retiradas
# Reajuste anual dos aportes: o valor nominal é corrigido em janeiro pelo IPCA acumulado de janeiro a dezembro
# do ano anterior e fica fixo até dezembro. Em poder de compra de hoje (preços do início do 1º ano), o aporte
# do k-ésimo mês de cada ano vale P / (1 + IPCA)^(k/12): perde valor real ao longo do ano e o recupera em janeiro.
def perfil(pi=IPCA):
    return (1 + pi) ** (-np.arange(1, 13) / 12)
PERFIL = perfil()
def fator_ap(rm, meses=MESES, perf=None):
    """Patrimônio final, em poder de compra de hoje, de aportes de R$ 1 (a preços do início de cada ano)
    feitos ao fim de cada mês, sob o reajuste anual, capitalizados à taxa real mensal rm."""
    perf = PERFIL if perf is None else perf
    m = np.arange(1, meses + 1)
    return float(np.sum(perf[(m - 1) % 12] * (1 + rm) ** (meses - m)))
contrib = APORTE * PERFIL.sum() * MESES / 12                   # total aportado em poder de compra de hoje
contrib_nom = APORTE * MESES                                    # o mesmo total a preços do início de cada ano
R['janelas'] = {'eq70_nom': eq70_nom, 'eq70_real': eq70_real, 'eq70_vol': eq70_vol,
                'eq94_nom': eq94_nom, 'eq94_real': eq94_real, 'eq94_vol': eq94_vol,
                'fx_tend': fx_tend, 'fx94_nom': fx94_nom, 'ptax_ini': 1.00, 'ptax_fim': float(ptax_m['2025-06']),
                'ipca94': ipca94, 'cpi94': cpi94, 'otimista': OTIMISTA,
                'parcela_cambio': np.log(1 + fx_tend) / np.log(1 + OTIMISTA), 'fx_ic95': fx_ic, 'fx_adf': adf}
print(f"=== JANELAS: MSCI World Net 27/02/1970–27/02/2025 {eq70_nom:.2%} nominal | {eq70_real:.2%} real | vol {eq70_vol:.2%}")
print(f"    01/07/1994–01/07/2025: ações {eq94_nom:.2%} nominal | {eq94_real:.2%} real | vol {eq94_vol:.2%} | "
      f"câmbio nominal {fx94_nom:.2%} | tendência real {fx_tend:.2%} | IPCA {ipca94:.2%} | CPI {cpi94:.2%}")
print(f"    otimista (período do real) {OTIMISTA:.2%} | parcela do câmbio {R['janelas']['parcela_cambio']:.0%}")
print("    IC 95% da tendência cambial (Newey-West): " + ' | '.join(f"{L} meses: {a:.2%} a {b:.2%}" for L, (a, b) in fx_ic.items())
      + f" | ADF {adf:.2f} (crítico 5% ≈ −3,42)")

# cenários de retorno REAL em R$ (antes do TER, deduzido aqui):
# conservador 3%; base 5% (≈ MSCI World Net real desde 1970) com câmbio neutro; otimista = período do real
CEN = {'Conservador': 0.03 - TER, 'Base': RET_BASE - TER, 'Otimista': OTIMISTA - TER}

# ---------------- 3. ACUMULAÇÃO (3 cenários, determinística) ----------------
print("=== ACUMULAÇÃO 30 anos (R$300/mês real; aportado R$ %s) ===" % f"{contrib:,.0f}")
acum = {}
for n, r in CEN.items():
    rm = mensal(r); fv = ap_liq * fator_ap(rm); acum[n] = fv
    print(f"  {n:24s} r={r:7.4%}  patrimônio R$ {fv:>10,.0f}")
W0 = acum['Base']
R['acumulacao'] = acum
os.makedirs('saidas', exist_ok=True)
rmb = mensal(CEN['Base']); s = 0.0; serie = []
for m in range(MESES):
    s = s * (1 + rmb) + ap_liq * PERFIL[m % 12]; serie.append(s)
# efeito do reajuste anual frente ao reajuste mensal (aporte de poder de compra constante), no cenário base
W_mensal = ap_liq * ((1 + rmb) ** MESES - 1) / rmb
W_anual3 = ap_liq * fator_ap(rmb, perf=perfil(0.03))
R['reajuste'] = {'ipca': IPCA, 'contrib': contrib, 'contrib_nom': contrib_nom, 'W_mensal': W_mensal,
                 'efeito': W0 / W_mensal - 1, 'efeito_3': W_anual3 / W_mensal - 1}
print(f"  reajuste anual (IPCA {IPCA:.2%}): {W0 / W_mensal - 1:+.2%} frente ao reajuste mensal (R$ {W_mensal:,.0f}); "
      f"com inflação de 3%: {W_anual3 / W_mensal - 1:+.2%} | total aportado R$ {contrib:,.0f} em poder de compra de hoje "
      f"(R$ {contrib_nom:,.0f} a preços do início de cada ano)")
pd.DataFrame({'mes': range(1, MESES + 1), 'saldo_real': serie}).to_csv('saidas/acumulacao_base.csv', index=False)

# ---------------- 4. MONTE CARLO — ACUMULAÇÃO ----------------
rng = np.random.default_rng(42)
fin = np.array([acumula(blocos(real_mc, 30, rng)) for _ in range(N)])
p5, p50, p95 = np.percentile(fin, [5, 50, 95])
print(f"=== MONTE CARLO acumulação (10.000): p5 R$ {p5:,.0f} | mediana R$ {p50:,.0f} | p95 R$ {p95:,.0f}")
R['mc_acum'] = {'p5': p5, 'p50': p50, 'p95': p95, 'acima_aportado': float((fin > contrib).mean())}

# ---------------- 5. JANELAS HISTÓRICAS — ACUMULAÇÃO ----------------
fins = []
for st in range(1970, 1996):
    seq = real_hist[(anos_arr >= st) & (anos_arr <= st + 29)]
    if len(seq) == 30:
        fins.append((st, acumula(seq)))
fdf = pd.DataFrame(fins, columns=['ini', 'pat'])
print(f"=== JANELAS acumulação ({len(fdf)}): pior R$ {fdf.pat.min():,.0f} ({int(fdf.loc[fdf.pat.idxmin(),'ini'])}) | "
      f"mediana R$ {fdf.pat.median():,.0f} | melhor R$ {fdf.pat.max():,.0f} ({int(fdf.loc[fdf.pat.idxmax(),'ini'])})")

# ---------------- 6. REGRAS DE RETIRADA ----------------
# Retirada anual no início de cada ano; renda mensal = retirada / 12.
def din(W0, ret, taxa, ter=TER):
    saldo = W0; ga = W0 * taxa; g_, s_ = [], []
    for i, r in enumerate(ret):
        g = W0 * taxa if i == 0 else min(ga * (1 + TETO), max(ga * (1 + PISO), saldo * taxa))
        g = min(g, saldo); saldo = (saldo - g) * (1 + liq_ter(r, ter)); ga = g
        g_.append(g); s_.append(max(saldo, 0))
        if saldo <= 0:
            g_ += [0] * (len(ret) - i - 1); s_ += [0] * (len(ret) - i - 1); break
    return np.array(g_), np.array(s_)

def fixa(W0, ret, taxa, ter=TER):
    saldo = W0; g0 = W0 * taxa; g_, s_ = [], []
    for r in ret:
        g = min(g0, saldo); saldo = (saldo - g) * (1 + liq_ter(r, ter))
        g_.append(g); s_.append(max(saldo, 0))
        if saldo <= 0: break
    while len(g_) < len(ret): g_.append(0); s_.append(0)
    return np.array(g_), np.array(s_)

def mc_regra(func, taxa, serie=real_mc, bloco=BLOCK, anos=ANOS_RET):
    rng2 = np.random.default_rng(42); ru = 0; rd = []
    for _ in range(N):
        g, s = func(W0, blocos(serie, anos, rng2, bloco), taxa)
        if s[-1] <= 0: ru += 1
        rd.append(g[g > 0].mean() / 12 if (g > 0).any() else 0)
    return ru / N, float(np.median(rd))

print("=== REGRAS DE RETIRADA (patrimônio base R$ %s) ===" % f"{W0:,.0f}")
R['regras'] = {}
for lbl, f, t in [('Dinâmica 4%', din, .04), ('Dinâmica 5%', din, .05), ('Fixa 4%', fixa, .04), ('Fixa 5%', fixa, .05)]:
    ru, rd = mc_regra(f, t); R['regras'][lbl] = {'ruina': ru, 'renda': rd}
    print(f"  {lbl:12s}: ruína {ru:5.1%} | renda mediana R$ {rd:,.0f}")
print("=== CURVA SWR (dinâmica, 7 pontos) ===")
R['swr'] = {}
for t in (.03, .035, .04, .045, .05, .055, .06):
    ru, rd = mc_regra(din, t); R['swr'][f'{t:.3f}'] = {'ruina': ru, 'renda': rd}
    print(f"  taxa {t:.1%}: ruína {ru:5.1%} | renda R$ {rd:,.0f}")
ru4 = R['regras']['Dinâmica 4%']['ruina']; ep = np.sqrt(ru4 * (1 - ru4) / N)
R['ic95_ruina'] = [ru4 - 1.96 * ep, ru4 + 1.96 * ep]
print(f"  IC 95% da ruína (dinâmica 4%): {R['ic95_ruina'][0]:.2%} a {R['ic95_ruina'][1]:.2%}")

# Cauda da renda (mesmas trajetórias): percentil 10 da renda mensal média e proporção de
# trajetórias em que a renda de algum ano cai abaixo de 75% da renda inicial.
def mc_cauda(func, taxa, serie=real_mc, bloco=BLOCK):
    rng2 = np.random.default_rng(42); medias, corte = [], 0
    for _ in range(N):
        g, s = func(W0, blocos(serie, ANOS_RET, rng2, bloco), taxa)
        medias.append(g.mean() / 12)
        corte += bool((g < 0.75 * g[0]).any())
    return float(np.percentile(medias, 10)), corte / N
for lbl, f, t in [('Dinâmica 4%', din, .04), ('Dinâmica 5%', din, .05), ('Fixa 4%', fixa, .04), ('Fixa 5%', fixa, .05)]:
    p10, ct = mc_cauda(f, t); R['regras'][lbl].update({'renda_p10': p10, 'corte25': ct})
    print(f"  {lbl:12s}: renda média P10 R$ {p10:,.0f} | renda abaixo de 75% da inicial em algum ano: {ct:.1%}")

# Regra dinâmica com proteções opcionais (usada nas seções 12 e 15):
def din_prot(W0, ret, taxa, piso=PISO, reserva=0, rc=0.0, ter=TER):
    caixa = reserva * W0 * taxa; saldo = W0 - caixa; ga = W0 * taxa; g_, s_ = [], []; prev = 0.0
    for i, r in enumerate(ret):
        tot = saldo + caixa
        g = W0 * taxa if i == 0 else min(ga * (1 + TETO), max(ga * (1 + piso), tot * taxa))
        g = min(g, tot)
        if reserva and prev < 0 and caixa > 0:
            dc = min(g, caixa); caixa -= dc; saldo -= g - dc
        else:
            saldo -= g
            if reserva and prev >= 0:
                mov = min(max(0, reserva * g - caixa), max(saldo, 0)); caixa += mov; saldo -= mov
        saldo *= 1 + liq_ter(r, ter); caixa *= 1 + rc; ga = g; prev = r
        g_.append(g); s_.append(max(saldo + caixa, 0))
        if saldo + caixa <= 0:
            g_ += [0] * (len(ret) - i - 1); s_ += [0] * (len(ret) - i - 1); break
    return np.array(g_), np.array(s_)

# ---------------- 7. RISCO DE SEQUÊNCIA (usufruto, janelas reais) ----------------
seq = []
for st in range(1970, 1996):
    s30 = real_hist[(anos_arr >= st) & (anos_arr <= st + 29)]
    if len(s30) == 30:
        g, s = din(W0, s30, 0.04); seq.append((st, s[-1], g[g > 0].mean() / 12, s[-1] <= 0))
sdf = pd.DataFrame(seq, columns=['ini', 'saldo', 'renda', 'ruina'])
pior = sdf.loc[sdf.saldo.idxmin()]; melhor = sdf.loc[sdf.saldo.idxmax()]
R['sequencia'] = {'ruinas': int(sdf.ruina.sum()), 'pior_ini': int(pior.ini), 'pior_saldo': pior.saldo,
                  'pior_renda': pior.renda, 'melhor_ini': int(melhor.ini), 'melhor_saldo': melhor.saldo,
                  'melhor_renda': melhor.renda}
fdf_r = {'pior': fdf.pat.min(), 'pior_ini': int(fdf.loc[fdf.pat.idxmin(), 'ini']), 'mediana': fdf.pat.median(),
         'melhor': fdf.pat.max(), 'melhor_ini': int(fdf.loc[fdf.pat.idxmax(), 'ini'])}
R['janelas_acum'] = fdf_r
print(f"=== SEQUÊNCIA usufruto ({len(sdf)} janelas, ruínas {int(sdf.ruina.sum())}): "
      f"pior {int(pior.ini)} saldo R$ {pior.saldo:,.0f}/renda R$ {pior.renda:,.0f} | "
      f"melhor {int(melhor.ini)} saldo R$ {melhor.saldo:,.0f}/renda R$ {melhor.renda:,.0f}")

# ---------------- 8. APORTE x INSS ----------------
print("=== SENSIBILIDADE DE APORTES x INSS (base) ===")
R['aportes'] = {}
for ap in (300, 500, 1000, 2000):
    fv = ap * (1 - C_APORTE) * fator_ap(rmb); rd = fv * 0.04 / 12; rd3 = fv * 0.03 / 12
    R['aportes'][str(ap)] = {'patrimonio': fv, 'renda': rd, 'renda_3': rd3}
    print(f"  R$ {ap:>5}/mês: patrimônio R$ {fv:>10,.0f} | renda a 4% R$ {rd:>5,.0f} ({rd/INSS_MEDIO*100:.0f}% do INSS médio)"
          f" | a 3% R$ {rd3:>5,.0f} ({rd3/INSS_MEDIO*100:.0f}%)")
ap_meta = INSS_MEDIO * 12 / 0.04 / fator_ap(rmb) / (1 - C_APORTE)
print(f"  -> aporte para renda bruta equivalente ao INSS médio (R$ {INSS_MEDIO:,.0f}): R$ {ap_meta:,.0f}/mês a 4%"
      f" | R$ {ap_meta * 0.04 / 0.035:,.0f} a 3,5% | R$ {ap_meta * 0.04 / 0.03:,.0f} a 3%")

# ---------------- 9. TRAJETÓRIA DA RENDA (dinâmica 4%) ----------------
rng3 = np.random.default_rng(42); paths = np.zeros((N, ANOS_RET))
for k in range(N):
    g, s = din(W0, blocos(real_mc, ANOS_RET, rng3), 0.04); paths[k] = g / 12
p10, p50t, p90 = np.percentile(paths, [10, 50, 90], axis=0)
print(f"=== TRAJETÓRIA renda dinâmica 4%: ano 1 mediana R$ {p50t[0]:,.0f} | ano 30 mediana R$ {p50t[-1]:,.0f}")
R['trajetoria'] = {'ano1': p50t[0], 'ano30': p50t[-1]}

# ---------------- 10. SWRD x BURT39 (cenário base; a renda líquida de IR está na seção 14) ----------------
# BURT39 = BDR do iShares MSCI World ETF (URTH), domiciliado nos EUA e distribuidor:
# seus dividendos sofrem retenção de 30% nos EUA (sem tratado Brasil–EUA) e taxa de custódia
# do BDR; o SWRD, irlandês, sofre 15% sobre dividendos americanos no fundo (tratado Irlanda–EUA).
DY, PESO_EUA = 0.0138, 0.727                     # rendimento distribuído do URTH; peso dos EUA
PERDA_DIV = DY * (0.30 + 0.03) - PESO_EUA * DY * 0.15
def fv_base(r, aporte=APORTE):
    rm = mensal(r); return aporte * fator_ap(rm)
# Na acumulação não há IR sobre o ganho de capital (ETF de acumulação e BDR mantidos sem venda): o patrimônio
# comparado é o acumulado; o IR incide só sobre o ganho nominal de cada venda de cotas, na fase de retiradas.
R_BURT = 0.05 - 0.0024 - PERDA_DIV                # TER 0,24% + perda com dividendos
W_swrd = W0                                       # já líquido do custo de entrada
W_burt = fv_base(R_BURT)
print(f"=== SWRD x BURT39 (perda anual com dividendos no BURT39: {PERDA_DIV:.2%}) ===")
print(f"  patrimônio em 30 anos, antes do IR sobre o ganho de capital: SWRD R$ {W_swrd:,.0f} | BURT39 R$ {W_burt:,.0f} | diferença {W_swrd/W_burt-1:+.1%}")
R['swrd_burt'] = {'perda_div': PERDA_DIV, 'swrd_pat': W_swrd, 'burt_pat': W_burt}

# ---------------- 11. AÇÕES EM R$ REAIS: ANOS DE JULHO A JUNHO, 01/07/1994 A 01/07/2025 ----------------
anos_brl, eq_usd, fxv, ipca_a = [], [], [], []
for k in range(31):                                           # 1994/95 a 2024/25
    ini, fim = f'{1994 + k}-06', f'{1995 + k}-06'
    ms_ = [m for m in meses94 if ini < m <= fim]              # julho a junho
    anos_brl.append(1995 + k)
    eq_usd.append(msci_m[fim] / msci_m[ini] - 1)
    fxv.append(ptax_m[fim] / (1.00 if k == 0 else ptax_m[ini]) - 1)
    ipca_a.append(np.prod(1 + ipca_mm[ms_].values) - 1)
eq_usd, fxv, ipca_a = map(np.array, (eq_usd, fxv, ipca_a))
acao_brl = (1 + eq_usd) * (1 + fxv) / (1 + ipca_a) - 1        # ações em R$ reais (TER aplicado na simulação)
def cagr(x): return np.prod(1 + x) ** (1 / len(x)) - 1
# Volatilidade em reais x em dólares: ações em USD reais (CPI) e componente cambial real
cpi_a = np.array([cpi_m[f'{1995 + k}-06'] / cpi_m[f'{1994 + k}-06'] - 1 for k in range(31)])
eq_usd_real = (1 + eq_usd) / (1 + cpi_a) - 1
cambio_real = (1 + acao_brl) / (1 + eq_usd_real) - 1
R['vol_brl'] = {'acao_brl': float(acao_brl.std(ddof=1)), 'acao_usd': float(eq_usd_real.std(ddof=1)),
                'cambio_real': float(cambio_real.std(ddof=1)), 'corr': float(np.corrcoef(eq_usd_real, cambio_real)[0, 1]),
                # "hedge natural": variação nominal média do dólar nos anos de queda e de alta das ações em dólares
                'n_queda': int((eq_usd < 0).sum()), 'fx_queda': float(fxv[eq_usd < 0].mean()), 'fx_alta': float(fxv[eq_usd >= 0].mean())}
print("=== VOLATILIDADE (jul.→jun., 1994/95–2024/25): ações em R$ reais {acao_brl:.1%} | em USD reais {acao_usd:.1%} "
      "| câmbio real {cambio_real:.1%} | correlação ações x câmbio {corr:.2f}".format(**R['vol_brl']))
print("    nos {n_queda} anos de queda das ações em dólares, o dólar variou em média {fx_queda:+.1%} (nos demais, {fx_alta:+.1%})".format(**R['vol_brl']))

# ---------------- 11b. AÇÕES BRASILEIRAS (IBOVESPA) X AÇÕES GLOBAIS, EM R$ REAIS, NO PERÍODO DO REAL ----------------
# Ibovespa (índice de retorno total da B3): fechamento do último pregão de cada mês, em pontos ajustados pelas
# divisões do índice por 10 de 10/02/1994 e 03/03/1997. Mesma janela (01/07/1994 a 01/07/2025), mesmos anos de
# julho a junho e mesmo deflator (IPCA) da série das ações globais convertida em reais.
ibov_m = pd.read_csv('dados/ibovespa_mensal.csv').set_index('mes')['pontos_ajustados']
ibov_a = np.array([ibov_m[f'{1995 + k}-06'] / ibov_m[f'{1994 + k}-06'] - 1 for k in range(31)])
ibov_brl = (1 + ibov_a) / (1 + ipca_a) - 1                      # Ibovespa em R$ reais
meses_b = ['1994-06'] + meses94
niv_ipca_b = np.r_[1.0, nivel_ipca]                              # IPCA: fim de jun./1994 = 1
niv_glob = np.r_[1.0, msci_m[meses94].values / msci_m['1994-06'] * ptax_m[meses94].values / 1.00] / niv_ipca_b
niv_ibov = ibov_m[meses_b].values / ibov_m['1994-06'] / niv_ipca_b
def queda_max(nivel):
    """Maior queda real entre um pico e o vale seguinte (dados mensais), com as datas e a da recuperação do pico."""
    pico = np.maximum.accumulate(nivel); v = int(np.argmin(nivel / pico)); p = int(np.argmax(nivel[:v + 1]))
    rec = next((meses_b[j] for j in range(v, len(nivel)) if nivel[j] >= nivel[p]), 'não recuperado até jun./2025')
    return float(nivel[v] / nivel[p] - 1), meses_b[p], meses_b[v], rec
misto = 0.5 * acao_brl + 0.5 * ibov_brl                           # 50% em cada, rebalanceada a cada julho
R['ibov'] = {'cagr_glob': cagr(acao_brl), 'cagr_ibov': cagr(ibov_brl), 'final_glob': 100 * niv_glob[-1], 'final_ibov': 100 * niv_ibov[-1],
             'vol_glob': float(acao_brl.std(ddof=1)), 'vol_ibov': float(ibov_brl.std(ddof=1)),
             'dd_glob': queda_max(niv_glob), 'dd_ibov': queda_max(niv_ibov),
             'pior_glob': [float(acao_brl.min()), int(anos_brl[int(acao_brl.argmin())])], 'pior_ibov': [float(ibov_brl.min()), int(anos_brl[int(ibov_brl.argmin())])],
             'anos_neg_glob': int((acao_brl < 0).sum()), 'anos_neg_ibov': int((ibov_brl < 0).sum()),
             'corr': float(np.corrcoef(acao_brl, ibov_brl)[0, 1]), 'anos_ibov_vence': int((ibov_brl > acao_brl).sum()),
             'misto': [cagr(misto), float(misto.std(ddof=1))]}
print("=== IBOVESPA x AÇÕES GLOBAIS EM R$ REAIS (01/07/1994–01/07/2025): retorno real {:.2%} x {:.2%} | R$ 100 → R$ {:.0f} x R$ {:.0f} "
      "| volatilidade {:.1%} x {:.1%} | correlação {:.2f}".format(R['ibov']['cagr_ibov'], R['ibov']['cagr_glob'], R['ibov']['final_ibov'],
      R['ibov']['final_glob'], R['ibov']['vol_ibov'], R['ibov']['vol_glob'], R['ibov']['corr']))
print("    queda máxima real: Ibovespa {:.1%} ({} a {}; pico recuperado: {}) | global {:.1%} ({} a {}; pico recuperado: {})".format(*R['ibov']['dd_ibov'], *R['ibov']['dd_glob']))
print("    Ibovespa superior em {} de 31 anos | 50%/50%: {:.2%} (volatilidade {:.1%})".format(R['ibov']['anos_ibov_vence'], *R['ibov']['misto']))

# ---------------- 11c. CDI X AÇÕES GLOBAIS, EM R$ REAIS, NO PERÍODO DO REAL ----------------
# CDI: taxa DI acumulada no mês (série 4391 do SGS do Banco Central). Mesma janela, mesmos anos de julho a junho e
# mesmo deflator (IPCA). Aplicação única: IR de 15% sobre o ganho nominal no resgate final nos dois casos (renda fixa
# mantida por mais de 720 dias; ações globais, Lei nº 14.754/2023), com custos de conversão (2,10% na ida, 1,38% na
# volta) e TER de 0,12% a.a. nas ações globais. Aportes: R$ 300 mensais em valores reais (R$ de jun./2025), ao fim de
# cada mês, de jul./1995 a jun./2025 (360 meses), antes do IR; custo de conversão e TER só nas ações globais.
cdi_mm = pd.read_csv('dados/cdi_mensal.csv'); cdi_mm = cdi_mm.set_index(cdi_mm.data.str[:7])['valor'] / 100
cdi_a = np.array([np.prod(1 + cdi_mm[[m for m in meses94 if f'{1994 + k}-06' < m <= f'{1995 + k}-06']].values) - 1 for k in range(31)])
cdi_brl = (1 + cdi_a) / (1 + ipca_a) - 1                         # CDI em R$ reais
niv_cdi = np.r_[1.0, np.cumprod(1 + cdi_mm[meses94].values)] / niv_ipca_b
usd = lambda x: cagr((1 + x) / (1 + fxv) - 1)                     # retorno nominal em dólares
P_ipca = np.prod(1 + ipca_a)
v_cdi = np.prod(1 + cdi_a); liq_cdi = v_cdi - IR * (v_cdi - 1)
v_glob = (1 - C_APORTE) * np.prod((1 + eq_usd) * (1 + fxv) * (1 - TER))
liq_glob = v_glob - IR * max(v_glob - (1 - C_APORTE), 0) - C_SAIDA * v_glob
def acum_real(r_mes, custo=0.0, ter=0.0):
    s_ = 0.0
    for r_ in r_mes: s_ = s_ * (1 + r_) * (1 - ter) ** (1 / 12) + APORTE * (1 - custo)
    return s_
m95 = [m for m in meses94 if m >= '1995-07']; ant = {m: meses_b[meses_b.index(m) - 1] for m in m95}
rg = np.array([msci_m[m] / msci_m[ant[m]] * ptax_m[m] / ptax_m[ant[m]] / (1 + ipca_mm[m]) - 1 for m in m95])
rc = np.array([(1 + cdi_mm[m]) / (1 + ipca_mm[m]) - 1 for m in m95])
ri = np.array([ibov_m[m] / ibov_m[ant[m]] / (1 + ipca_mm[m]) - 1 for m in m95])
R['cdi'] = {'cagr_cdi': cagr(cdi_brl), 'final_cdi': 100 * niv_cdi[-1], 'vol_cdi': float(cdi_brl.std(ddof=1)),
            'pior_cdi': [float(cdi_brl.min()), int(anos_brl[int(cdi_brl.argmin())])], 'melhor_cdi': [float(cdi_brl.max()), int(anos_brl[int(cdi_brl.argmax())])],
            'anos_neg_cdi': int((cdi_brl < 0).sum()), 'dd_cdi': queda_max(niv_cdi),
            'anos_cdi_vence': int((cdi_brl > acao_brl).sum()), 'corr_glob': float(np.corrcoef(acao_brl, cdi_brl)[0, 1]),
            'usd_cdi': usd(cdi_a), 'usd_glob': cagr(eq_usd), 'usd_ibov': usd(ibov_a), 'cdi_nominal': cagr(cdi_a),
            'liq_cdi': (liq_cdi / P_ipca) ** (1 / 31) - 1, 'liq_glob': (liq_glob / P_ipca) ** (1 / 31) - 1,
            'ap_cdi': acum_real(rc), 'ap_glob': acum_real(rg, C_APORTE, TER), 'ap_glob_sem_custo': acum_real(rg), 'ap_ibov': acum_real(ri),
            'total_aportado': APORTE * len(m95)}
print("=== CDI x AÇÕES GLOBAIS EM R$ REAIS (01/07/1994–01/07/2025): retorno real {cagr_cdi:.2%} | R$ 100 → R$ {final_cdi:.0f} "
      "| volatilidade {vol_cdi:.1%} | anos negativos {anos_neg_cdi} | CDI superior em {anos_cdi_vence} de 31 anos | correlação {corr_glob:.2f}".format(**R['cdi']))
print("    em dólares (nominal): CDI {usd_cdi:.2%} | MSCI World Net {usd_glob:.2%} | Ibovespa {usd_ibov:.2%}".format(**R['cdi']))
print("    aplicação única após IR (e custos da estratégia): CDI {liq_cdi:.2%} x global {liq_glob:.2%} reais a.a.".format(**R['cdi']))
print("    aportes de R$ 300 (jul./1995–jun./2025, antes do IR): CDI R$ {ap_cdi:,.0f} | global R$ {ap_glob:,.0f} (sem custos R$ {ap_glob_sem_custo:,.0f}) "
      "| Ibovespa R$ {ap_ibov:,.0f} | aportado R$ {total_aportado:,.0f}".format(**R['cdi']))

# ---------------- 12. CÂMBIO COMO VARIÁVEL ALEATÓRIA ----------------
# Mesmo método da seção 4, mas sorteando blocos de anos (julho a junho) em que ações, câmbio e IPCA
# permanecem pareados como ocorreram no período do real. Duas versões: sem tendência (série em R$
# deslocada até a média do cenário base, de modo que o câmbio acrescente apenas dispersão) e com a
# tendência de depreciação real do real medida no período (base × tendência cambial).
acao_brl_c, D_CB = recentra(acao_brl)
ALVO_TEND = (1 + RET_BASE) * (1 + fx_tend) - 1
acao_brl_t, D_CT = recentra(acao_brl, ALVO_TEND)
R['deflator'].update({'geo_acao_brl': geo(acao_brl), 'desloc_cambio': D_CB, 'alvo_tend': ALVO_TEND, 'desloc_tend': D_CT})
R['cambio'] = {}
for chave, serie_ in (('sem_tendencia', acao_brl_c), ('tendencia_real', acao_brl_t)):
    rng4 = np.random.default_rng(42)
    f_ = np.array([acumula(blocos(serie_, 30, rng4)) for _ in range(N)])
    q5, q50, q95 = np.percentile(f_, [5, 50, 95])
    R['cambio'][chave] = {'p5': q5, 'p50': q50, 'p95': q95, 'acima_aportado': float((f_ > contrib).mean()),
                          'taxas': {}, 'flex': {}}
    print(f"=== CÂMBIO ALEATÓRIO ({chave}): p5 R$ {q5:,.0f} | mediana R$ {q50:,.0f} | p95 R$ {q95:,.0f} "
          f"| acima do aportado {(f_ > contrib).mean():.1%}")
    for t in (.03, .035, .04):
        ru, rd = mc_regra(din, t, serie=serie_); R['cambio'][chave]['taxas'][f'{t:.3f}'] = {'ruina': ru, 'renda': rd}
        print(f"  usufruto taxa {t:.1%}: ruína {ru:5.1%} | renda R$ {rd:,.0f}")
    R['cambio'][chave]['bloco10'] = {f'{t:.3f}': mc_regra(din, t, serie=serie_, bloco=10)[0] for t in (.035, .04)}
    for pz in (-0.05, -0.10):
        for t in (.035, .04):
            ru, rd = mc_regra(lambda W, ret, tx: din_prot(W, ret, tx, piso=pz), t, serie=serie_)
            R['cambio'][chave]['flex'][f'{pz:.2f}_{t:.3f}'] = {'ruina': ru, 'renda': rd}
    print(f"  blocos de 10 anos: {R['cambio'][chave]['bloco10']} | flexibilidade: "
          + ', '.join(f"{k}: {v['ruina']:.1%}" for k, v in R['cambio'][chave]['flex'].items()))
    if chave == 'sem_tendencia': fin_c = f_
    else: fin_t = f_

# ---------------- 13. SENSIBILIDADE DAS PREMISSAS ----------------
print("=== SENSIBILIDADE ===")
R['sens'] = {'custo': {}, 'ter': {}, 'bloco': {}}
for c in (0.0150, 0.0310):                             # spread de 0,4% e de 2,0% + IOF de 1,10%
    v = APORTE * (1 - c) * fator_ap(mensal(0.05 - TER))
    R['sens']['custo'][f'{c:.4f}'] = v; print(f"  custo {c:.2%}: R$ {v:,.0f} ({v/W0-1:+.1%})")
for t in (0.0007, 0.0025):
    v = ap_liq * fator_ap(mensal(0.05 - t))
    R['sens']['ter'][f'{t:.4f}'] = v; print(f"  TER {t:.2%}: R$ {v:,.0f} ({v/W0-1:+.1%})")
# combinações extremas de custo de conversão e TER
for c, t in ((0.0150, 0.0007), (0.0310, 0.0025)):
    v = APORTE * (1 - c) * fator_ap(mensal(0.05 - t))
    R['sens'].setdefault('extremos', {})[f'{c:.4f}_{t:.4f}'] = v; print(f"  custo {c:.2%} e TER {t:.2%}: R$ {v:,.0f} ({v/W0-1:+.1%})")
# corretagem por ordem na Bolsa de Londres (fora do cenário base): tarifa mínima em USD convertida pelo câmbio
# de jun./2025; compras a cada k meses, com o caixa aguardando a compra sem rendimento real
def fv_ordens(tarifa_usd, k):
    rm = mensal(0.05 - TER); tarifa = tarifa_usd * ptax_m['2025-06']; saldo = caixa = 0.0
    for m in range(1, MESES + 1):
        saldo *= 1 + rm; caixa += ap_liq * PERFIL[(m - 1) % 12]
        if m % k == 0: saldo += caixa - tarifa; caixa = 0.0
    return saldo + caixa
assert abs(fv_ordens(0, 1) / W0 - 1) < 1e-9
R['sens']['corretagem'] = {'aporte_usd': ap_liq / ptax_m['2025-06']}
for tar, k in ((1.70, 1), (4.00, 1), (1.70, 3), (4.00, 3)):
    v = fv_ordens(tar, k); R['sens']['corretagem'][f'{tar:.2f}_{k}'] = v
    print(f"  corretagem US$ {tar:.2f} por ordem, compras a cada {k} mês(es): R$ {v:,.0f} ({v/W0-1:+.1%}) "
          f"| tarifa = {tar * ptax_m['2025-06'] / (ap_liq * k):.1%} da ordem")
for b in (1, 3, 10):
    ru, _ = mc_regra(din, 0.04, bloco=b); ru35, _ = mc_regra(din, 0.035, bloco=b)
    R['sens']['bloco'][str(b)] = {'0.040': ru, '0.035': ru35}
    print(f"  bloco {b:>2} anos: ruína {ru:.1%} (4%) | {ru35:.1%} (3,5%)")
# retornos futuros inferiores ao cenário base: desloca-se toda a série real em -1 e -2 p.p.
R['sens']['retorno'] = {}
for d in (0.01, 0.02):
    for t in (.04, .035):
        ru, rd = mc_regra(din, t, serie=real_mc - d)
        R['sens']['retorno'][f'-{d:.2f}_{t:.3f}'] = {'ruina': ru, 'renda': rd}
        print(f"  retorno -{d:.0%} p.p., taxa {t:.1%}: ruína {ru:.1%} | renda R$ {rd:,.0f}")

# Síntese da robustez: ruína de quatro parâmetros de retirada em todos os testes
flex5 = lambda W, ret, tx: din_prot(W, ret, tx, piso=-0.05)          # cortes anuais de até 5%
parametros = {'4%': (din, .04), '3,5%': (din, .035), '3%': (din, .03), '3,5% com cortes de até 5%': (flex5, .035)}
testes = {'Câmbio neutro (referência)': {}, 'Retorno 1 p.p. menor': {'serie': real_mc - 0.01},
          'Retorno 2 p.p. menor': {'serie': real_mc - 0.02}, 'Blocos de 1 ano': {'bloco': 1},
          'Blocos de 10 anos': {'bloco': 10}, 'Câmbio aleatório com tendência': {'serie': acao_brl_t},
          'Câmbio aleatório sem tendência': {'serie': acao_brl_c},
          'Câmbio aleatório sem tendência, blocos de 10 anos': {'serie': acao_brl_c, 'bloco': 10},
          'Câmbio aleatório sem tendência e retorno 1 p.p. menor': {'serie': acao_brl_c - 0.01}}
R['robustez'] = {}
print("=== SÍNTESE DA ROBUSTEZ (probabilidade de ruína em 30 anos) ===")
for teste, kw in testes.items():
    R['robustez'][teste] = {}
    for p, (f, t) in parametros.items():
        ru, rd = mc_regra(f, t, **kw); R['robustez'][teste][p] = {'ruina': ru, 'renda': rd}
    print(f"  {teste:55s} " + ' | '.join(f"{p}: {v['ruina']:5.1%}" for p, v in R['robustez'][teste].items()))
for p in parametros:
    ok = sum(R['robustez'][t][p]['ruina'] <= 0.10 for t in testes)
    print(f"  {p}: dentro do critério (ruína de até 10%) em {ok} de {len(testes)} testes")

# ---------------- 14. RENDA LÍQUIDA: IR NAS RETIRADAS E CUSTO DE SAÍDA ----------------
# Cada retirada exige a venda de cotas: incide IR de 15% sobre o ganho NOMINAL em reais
# (custo médio de aquisição; Lei nº 14.754/2023) e o custo de conversão de 1,38% na saída.
# Trajetória determinística do cenário base, regra dinâmica de 4%, inflação igual ao IPCA médio.
def renda_liquida(taxa=0.04, pi=IPCA, r=None, aporte=ap_liq, saida=C_SAIDA):
    r = CEN['Base'] if r is None else r; rm = mensal(r); s = 0.0; custo_nom = 0.0
    for t in range(MESES):
        s = s * (1 + rm) + aporte * PERFIL[t % 12]
        custo_nom += aporte * (1 + pi) ** (t // 12)       # custo de aquisição em R$ correntes (valor reajustado em janeiro)
    saldo = s; ga = s * taxa; bru, liq = [], []
    for k in range(ANOS_RET):
        g = s * taxa if k == 0 else min(ga * (1 + TETO), max(ga * (1 + PISO), saldo * taxa))
        P = (1 + pi) ** (MESES / 12 + k)                   # nível de preços no ano da venda
        frac = g / saldo
        ganho = g * P - custo_nom * frac
        custo_nom *= (1 - frac)
        bru.append(g); liq.append((g * P - IR * max(ganho, 0) - saida * g * P) / P)
        saldo = (saldo - g) * (1 + r); ga = g
    return np.array(bru), np.array(liq)
bru, liq = renda_liquida()
red = 1 - liq / bru
ap_meta_liq = ap_meta / (liq[0] / bru[0])
print(f"=== RENDA LÍQUIDA: redução de {red[0]:.1%} (ano 1) a {red[-1]:.1%} (ano 30); média {1-liq.mean()/bru.mean():.1%}")
print(f"  -> aporte para renda LÍQUIDA inicial equivalente ao INSS médio: R$ {ap_meta_liq:,.0f}/mês")
R['liquida'] = {'red_ano1': red[0], 'red_ano30': red[-1], 'red_media': 1 - liq.mean() / bru.mean(),
                'ap_meta_bruto': ap_meta, 'ap_meta_liq': ap_meta_liq}
# mesma meta à taxa de 3,5%
ap_meta35 = ap_meta * 0.04 / 0.035
bru35, liq35 = renda_liquida(0.035); ap_meta35_liq = ap_meta35 / (liq35[0] / bru35[0])
R['liquida'].update({'ap_meta_bruto_35': ap_meta35, 'ap_meta_liq_35': ap_meta35_liq})
print(f"  -> à taxa de 3,5%: aporte bruto R$ {ap_meta35:,.0f} | líquido R$ {ap_meta35_liq:,.0f}")
# mesma meta à taxa de 3%
ap_meta3 = ap_meta * 0.04 / 0.03
bru3, liq3 = renda_liquida(0.03); ap_meta3_liq = ap_meta3 / (liq3[0] / bru3[0])
R['liquida'].update({'ap_meta_bruto_3': ap_meta3, 'ap_meta_liq_3': ap_meta3_liq})
print(f"  -> à taxa de 3%: aporte bruto R$ {ap_meta3:,.0f} | líquido R$ {ap_meta3_liq:,.0f}")
# SWRD x BURT39 na fase de retiradas (regra dinâmica de 4%): IR de 15% sobre o ganho nominal de cada venda nos dois
# veículos; custo de conversão na saída só no SWRD; o BURT39 não tem custo de remessa na entrada nem na saída.
for nome, kw in (('swrd', {}), ('burt', dict(r=R_BURT, aporte=APORTE, saida=0.0))):
    b_, l_ = renda_liquida(**kw)
    R['swrd_burt'].update({f'{nome}_renda_ini': b_[0] / 12, f'{nome}_renda_liq_ini': l_[0] / 12,
                           f'{nome}_renda_liq_media': l_.mean() / 12})
sb_ = R['swrd_burt']
print(f"  -> SWRD x BURT39, renda líquida mensal inicial R$ {sb_['swrd_renda_liq_ini']:,.0f} x R$ {sb_['burt_renda_liq_ini']:,.0f}"
      f" | média do usufruto R$ {sb_['swrd_renda_liq_media']:,.0f} x R$ {sb_['burt_renda_liq_media']:,.0f}"
      f" ({sb_['swrd_renda_liq_media'] / sb_['burt_renda_liq_media'] - 1:+.1%})")
# Horizonte de equilíbrio: aportes mensais por H anos e resgate total ao fim, com IR de 15% sobre o ganho nominal nos dois
# veículos e custo de conversão na entrada e na saída só no SWRD (sem corretagem nem spread de compra e venda do BDR).
def liquido_resgate(anos, r, aporte, saida, pi=IPCA):
    rm = mensal(r); s_ = 0.0; custo_nom = 0.0
    for t in range(anos * 12):
        s_ = s_ * (1 + rm) + aporte * PERFIL[t % 12]
        custo_nom += aporte * (1 + pi) ** (t // 12)
    P = (1 + pi) ** anos
    return (s_ * P - IR * max(s_ * P - custo_nom, 0) - saida * s_ * P) / P
eq_ = [(h, liquido_resgate(h, CEN['Base'], ap_liq, C_SAIDA), liquido_resgate(h, R_BURT, APORTE, 0.0)) for h in range(1, 41)]
h_eq = next(h for h, a, b in eq_ if a >= b)
R['swrd_burt'].update({'horizonte_equilibrio': h_eq,
                       'resgate': {str(h): [a, b] for h, a, b in eq_ if h in (5, 10, h_eq - 1, h_eq, 20, 30)}})
print(f"  -> resgate total após H anos de aportes (líquido de IR e custos): SWRD supera o BURT39 a partir de {h_eq} anos | "
      + ' | '.join(f"{h} anos: R$ {a:,.0f} x R$ {b:,.0f}" for h, a, b in eq_ if h in (5, 10, h_eq - 1, h_eq, 20, 30)))
# esforço de poupança: aporte como proporção da renda do trabalho (PNAD Contínua 2024 e salário mínimo de 2025)
RENDA_MEDIA, SAL_MIN = 3225.0, 1518.0
R['acessibilidade'] = {k: {'renda_media': v / RENDA_MEDIA, 'salario_minimo': v / SAL_MIN} for k, v in
                       (('300', 300), ('4%', ap_meta), ('3,5%', ap_meta35), ('3%', ap_meta3),
                        ('4%_liq', ap_meta_liq), ('3,5%_liq', ap_meta35_liq), ('3%_liq', ap_meta3_liq))}
# Probabilidade de atingir a meta: o aporte central iguala a renda do INSS médio apenas quando o patrimônio
# chega ao do cenário base; como o patrimônio final é proporcional ao aporte, o aporte que assegura a meta
# com confiança c é o aporte central multiplicado por W0 / percentil (1 − c) da distribuição de Monte Carlo.
R['meta_confianca'] = {}
for nome, dist in (('neutro', fin), ('sem_tendencia', fin_c)):
    R['meta_confianca'][nome] = {'prob_central': float((dist >= W0).mean())}
    for c in (0.8, 0.9):
        fator = W0 / np.percentile(dist, 100 * (1 - c))
        R['meta_confianca'][nome][f'{c:.1f}'] = {'3%': ap_meta3 * fator, '3%_liq': ap_meta3_liq * fator, '3,5%': ap_meta35 * fator,
                                                 '3%_renda_media': ap_meta3 * fator / RENDA_MEDIA}
    print(f"  -> meta ({nome}): cenário central alcançado em {R['meta_confianca'][nome]['prob_central']:.1%} das trajetórias | "
          + ' | '.join(f"{c}: 3% R$ {v['3%']:,.0f} ({v['3%_renda_media']:.1%} da renda média), líquido R$ {v['3%_liq']:,.0f}, 3,5% R$ {v['3,5%']:,.0f}"
                       for c, v in R['meta_confianca'][nome].items() if c != 'prob_central'))
print("  -> aporte / renda média | / salário mínimo: " + ' ; '.join(
      f"{k}: {v['renda_media']:.1%} | {v['salario_minimo']:.1%}" for k, v in R['acessibilidade'].items()))

# ---------------- 15. PROTEÇÕES CONTRA O RISCO DE SEQUÊNCIA ----------------
# (a) reserva de liquidez: 'reserva' anos de retirada inicial em caixa (retorno real rc);
#     após ano de retorno negativo a retirada sai do caixa; após ano positivo o caixa é recomposto;
# (b) flexibilidade de gasto: piso de redução anual mais largo que -2,5%;
# (c) aposentadoria adiada: horizonte de usufruto menor e patrimônio maior (mais aportes).
print("=== PROTEÇÕES (dinâmica 4%) ===")
R['protecoes'] = {}
for lbl, kw, anos in [('Referência', {}, 30),
                      ('Reserva 1 ano (0% real)', dict(reserva=1), 30),
                      ('Reserva 2 anos (0% real)', dict(reserva=2), 30),
                      ('Reserva 2 anos (3% real)', dict(reserva=2, rc=0.03), 30),
                      ('Piso -5%', dict(piso=-0.05), 30),
                      ('Piso -10%', dict(piso=-0.10), 30),
                      ('Aposentadoria 2 anos depois', {}, 28),
                      ('Aposentadoria 5 anos depois', {}, 25)]:
    ru, rd = mc_regra(lambda W, ret, t: din_prot(W, ret, t, **kw), 0.04, anos=anos)
    m = MESES + (30 - anos) * 12                      # meses de aportes até a aposentadoria
    escala = fator_ap(rmb, m) / fator_ap(rmb)
    R['protecoes'][lbl] = {'ruina': ru, 'renda': rd * escala, 'patrimonio': W0 * escala}
    print(f"  {lbl:28s}: ruína {ru:5.1%} | renda mediana R$ {rd*escala:,.0f} | patrimônio R$ {W0*escala:,.0f}")

# ---------------- 15b. RUÍNA AO LONGO DA VIDA (TÁBUA DE MORTALIDADE DO IBGE, 2023) ----------------
# Probabilidade de o patrimônio esgotar-se enquanto o aposentado (ou ao menos um dos cônjuges) vive,
# para aposentadoria aos 65 anos. Os 30 primeiros anos de cada trajetória são os mesmos da seção 6
# (mesma semente); acrescentam-se 15 anos (até os 110) com outra semente. Após os 90 anos, supõe-se
# força de mortalidade constante, igual ao inverso da esperança de vida aos 90. Casal: homem e mulher
# de 65 anos, com tempos de vida independentes.
tab = pd.read_csv('dados/tabua_mortalidade_ibge_2023.csv').set_index('idade')
IDADE_APOS, H_VIDA = 65, 45
def sobrev(sexo, t):                                   # probabilidade de estar vivo aos 65 + t
    lx, e90 = tab[f'lx_{sexo}'], tab.loc[90, f'ex_{sexo}']
    x = IDADE_APOS + t
    l = lx[x] if x <= 90 else lx[90] * np.exp(-(x - 90) / e90)
    return l / lx[IDADE_APOS]
def ruina_em_vida(f, taxa, serie):
    rng2 = np.random.default_rng(42); rng3 = np.random.default_rng(7); anos = []
    for _ in range(N):
        ret = np.r_[blocos(serie, ANOS_RET, rng2), blocos(serie, H_VIDA - ANOS_RET, rng3)]
        _, s = f(W0, ret, taxa); z = np.flatnonzero(s <= 0)
        anos.append(z[0] + 1 if len(z) else None)
    r30 = np.mean([a is not None and a <= ANOS_RET for a in anos])
    ind = np.mean([sobrev('ambos', a) if a else 0 for a in anos])
    cas = np.mean([1 - (1 - sobrev('homens', a)) * (1 - sobrev('mulheres', a)) if a else 0 for a in anos])
    return r30, ind, cas
print("=== RUÍNA AO LONGO DA VIDA (aposentadoria aos 65; tábua IBGE 2023) ===")
R['vida'] = {'sobrevivencia': {str(x): {'individuo': sobrev('ambos', x - 65),
                                        'casal': 1 - (1 - sobrev('homens', x - 65)) * (1 - sobrev('mulheres', x - 65))}
                               for x in (85, 90, 95, 100)}}
for nome, serie_ in (('neutro', real_mc), ('sem_tendencia', acao_brl_c)):
    R['vida'][nome] = {}
    for p, (f, t) in parametros.items():
        r30, ind, cas = ruina_em_vida(f, t, serie_)
        R['vida'][nome][p] = {'ruina30': r30, 'individuo': ind, 'casal': cas}
        print(f"  {nome:14s} {p:26s}: ruína em 30 anos {r30:5.1%} | em vida: indivíduo {ind:5.1%} | casal {cas:5.1%}")
print("  sobrevivência a partir dos 65: " + ', '.join(f"{x} anos: {v['individuo']:.1%} (casal {v['casal']:.1%})"
                                                   for x, v in R['vida']['sobrevivencia'].items()))

# ---------------- 15b. ALTERNATIVAS DE ACUMULAÇÃO E HORIZONTE (Tabelas 15 e 16) ----------------
# Poupança: 0,45% real a.a. (média de 2012 a 2025); Tesouro RendA+: IPCA + 6,4% a.a. (taxa média desde o lançamento);
# ambos sem custo de conversão nem TER. Estratégia global: cenários base e otimista, como na seção 3.
R['veiculos'] = {'poupanca': fv_base(0.0045), 'rendamais': fv_base(0.064), 'base': W0, 'otimista': acum['Otimista']}
R['horizonte'] = {str(h): {'patrimonio': ap_liq * fator_ap(mensal(CEN['Base']), meses=12 * h), 'aportado': APORTE * PERFIL.sum() * h}
                  for h in (20, 25, 30, 35, 40)}
print("=== ALTERNATIVAS (R$ 300/mês, 30 anos): poupança R$ {poupanca:,.0f} | RendA+ R$ {rendamais:,.0f} | global base R$ {base:,.0f} "
      "| otimista R$ {otimista:,.0f}".format(**R['veiculos']))
print("=== HORIZONTE (cenário base): " + ' | '.join(f"{h} anos: R$ {v['patrimonio']:,.0f} (aportado R$ {v['aportado']:,.0f})"
                                                     for h, v in R['horizonte'].items()))
# Tesouro RendA+ (Tabelas 21 e 23): patrimônio acumulado à taxa real contratada e renda mensal por 20 anos (240 parcelas),
# calculada como anuidade à mesma taxa; para comparação, o patrimônio base da estratégia consumido em 20 anos a 5% reais.
def anuidade_20(W, r):
    rm = mensal(r); return W * rm / (1 - (1 + rm) ** -240)
R['rendamais'] = {f'{t:.3f}': {'patrimonio': fv_base(t), 'renda20': anuidade_20(fv_base(t), t)} for t in (0.07, 0.064, 0.04)}
R['rendamais']['consumo20_base'] = anuidade_20(W0, RET_BASE)
R['perfis_inss'] = {f'{c:.2f}': {'global_4': 0.04 * fv_base(CEN['Base'], aporte=c * (1 - C_APORTE)) / 12,
                                 'global_3': 0.03 * fv_base(CEN['Base'], aporte=c * (1 - C_APORTE)) / 12,
                                 'rendamais_20': anuidade_20(fv_base(0.064, aporte=c), 0.064)}
                    for c in (113.85, 280.41, 951.63)}          # desconto do INSS no piso, no rendimento médio e no teto (2025)
print("=== TESOURO RendA+: " + ' | '.join(f"IPCA + {float(t) * 100:.1f}%: patrimônio R$ {v['patrimonio']:,.0f}, renda 20 anos R$ {v['renda20']:,.0f}"
                                         for t, v in R['rendamais'].items() if t != 'consumo20_base')
      + f" | estratégia consumida em 20 anos a 5%: R$ {R['rendamais']['consumo20_base']:,.0f}")
print("=== PERFIS x INSS (aporte = desconto do INSS): " + ' | '.join(
      f"R$ {c}: global R$ {v['global_4']:,.0f} / R$ {v['global_3']:,.0f}; RendA+ R$ {v['rendamais_20']:,.0f}" for c, v in R['perfis_inss'].items()))

# ---------------- 16. ARQUIVOS PARA OS GRÁFICOS ----------------
np.save('saidas/mc_finais.npy', fin); np.save('saidas/mc_cambio.npy', fin_c); np.save('saidas/mc_cambio_tend.npy', fin_t)
pd.DataFrame({'ano': range(1, ANOS_RET + 1), 'p10': p10, 'p50': p50t, 'p90': p90}).to_csv('saidas/renda_traj.csv', index=False)
ip_, im_ = int(pior.ini), int(melhor.ini)
gp, sp = din(W0, real_hist[(anos_arr >= ip_) & (anos_arr <= ip_ + 29)], 0.04)
gm, sm = din(W0, real_hist[(anos_arr >= im_) & (anos_arr <= im_ + 29)], 0.04)
pd.DataFrame({'ano': range(1, ANOS_RET + 1), 'saldo_pior': sp, 'saldo_melhor': sm,
              'renda_pior': gp / 12, 'renda_melhor': gm / 12}).to_csv('saidas/sequencia.csv', index=False)
pd.DataFrame({'ano': anos_brl, 'acao_brl_real': acao_brl}).to_csv('saidas/series_reais_brl.csv', index=False)
# Apêndice B: séries anuais derivadas do período do real (julho a junho), em %
series_b = {'msci_usd': eq_usd, 'cpi_eua': cpi_a, 'dolar': fxv, 'ipca': ipca_a, 'msci_brl_real': acao_brl, 'ibov': ibov_a, 'ibov_brl_real': ibov_brl, 'cdi': cdi_a, 'cdi_brl_real': cdi_brl}
tab_b = pd.DataFrame({'ano': [f'{a - 1}/{str(a)[2:]}' for a in anos_brl], **{k: 100 * v for k, v in series_b.items()}})
tab_b.loc[len(tab_b)] = ['Média geométrica'] + [100 * cagr(v) for v in series_b.values()]   # com precisão integral
tab_b.to_csv('saidas/apendice_b_series.csv', index=False, float_format='%.2f')
# Apêndice C: retornos anuais do MSCI World Net desde o lançamento (fevereiro a fevereiro, 1970/71–2024/25), em %
inf_fev = cfev[1:] / cfev[:-1] - 1
tab_c = pd.DataFrame({'ano': [f'{a}/{str(a + 1)[2:]}' for a in anos_arr], 'msci_usd': 100 * nom, 'cpi_eua': 100 * inf_fev, 'real': 100 * real_hist})
tab_c.loc[len(tab_c)] = ['Média geométrica', 100 * geo(nom), 100 * geo(inf_fev), 100 * geo(real_hist)]
tab_c.to_csv('saidas/apendice_c_series.csv', index=False, float_format='%.2f')
pd.DataFrame({'mes': meses_b, 'global_real': 100 * niv_glob, 'ibov_real': 100 * niv_ibov, 'cdi_real': 100 * niv_cdi}).to_csv('saidas/ibov_global_mensal.csv', index=False)
with open('saidas/resultados.json', 'w') as fh:
    json.dump(R, fh, indent=1, default=float)
