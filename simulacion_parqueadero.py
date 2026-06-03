# -*- coding: utf-8 -*-
# Simulacion de Parqueaderos - Centro Comercial Supercentro
# Modelo M/M/1 - Simulacion por Eventos Discretos
# Alejandro Arango Calderon
# 5to Semestre - Modulo 4, Actividad Didactica 2

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from collections import defaultdict
import os
import sys

# Forzar UTF-8 en la consola de Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ==========================================
# CONFIGURACIÓN GLOBAL
# ==========================================
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURAS_DIR = os.path.join(OUTPUT_DIR, 'figuras')
os.makedirs(FIGURAS_DIR, exist_ok=True)

# Parámetros del modelo
TIPOS_USUARIO = ['Rápido', 'Normal', 'Lento', 'Muy Lento']
PROBABILIDADES = [0.25, 0.20, 0.275, 0.275]
MEDIA_SERVICIO = {'Rápido': 1, 'Normal': 3, 'Lento': 4, 'Muy Lento': 6}  # minutos
MEDIA_LLEGADA  = {'Rápido': 3, 'Normal': 3, 'Lento': 5, 'Muy Lento': 7}  # minutos

NUM_CAJEROS = 3
TIEMPO_SIMULACION = 480  # 8 horas en minutos
NUM_REPLICAS = 50

# Colores para las graficas
COLORES_TIPO = {
    'Rápido':    '#2ecc71',
    'Normal':    '#3498db',
    'Lento':     '#e67e22',
    'Muy Lento': '#e74c3c'
}
COLORES_CAJERO = ['#1abc9c', '#9b59b6', '#f39c12', '#e74c3c', '#3498db', '#2c3e50']

# Estilo global de gráficas
plt.rcParams.update({
    'figure.figsize': (12, 7),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
})


# --- FUNCIONES DE SIMULACION ---

def simular_cajero_mm1(tiempo_sim, semilla=None):
    """
    Simula un cajero M/M/1 con tipos de usuario mixtos.

    El proceso de llegada genera inter-llegadas exponenciales cuya
    media depende del tipo de usuario asignado aleatoriamente.
    El servicio es exponencial con media según el tipo de usuario.

    Retorna lista de dicts con datos de cada cliente atendido.
    """
    rng = np.random.RandomState(semilla)
    clientes = []
    tiempo = 0.0
    servidor_libre_en = 0.0

    while tiempo < tiempo_sim:
        # 1. Asignar tipo de usuario según probabilidades
        tipo_idx = rng.choice(len(TIPOS_USUARIO), p=PROBABILIDADES)
        tipo = TIPOS_USUARIO[tipo_idx]

        # 2. Inter-llegada ~ Exp(media_llegada[tipo])
        inter_llegada = rng.exponential(MEDIA_LLEGADA[tipo])
        tiempo += inter_llegada
        if tiempo >= tiempo_sim:
            break

        # 3. Servicio ~ Exp(media_servicio[tipo])
        tiempo_servicio = rng.exponential(MEDIA_SERVICIO[tipo])

        # 4. Determinar espera y tiempos
        if tiempo >= servidor_libre_en:
            espera = 0.0
            inicio_servicio = tiempo
        else:
            espera = servidor_libre_en - tiempo
            inicio_servicio = servidor_libre_en

        fin_servicio = inicio_servicio + tiempo_servicio
        servidor_libre_en = fin_servicio

        clientes.append({
            'llegada':         tiempo,
            'espera':          espera,
            'servicio':        tiempo_servicio,
            'en_sistema':      espera + tiempo_servicio,
            'tipo':            tipo,
            'salida':          fin_servicio,
            'inicio_servicio': inicio_servicio,
        })

    return clientes


def simular_sistema(num_cajeros, tiempo_sim, semilla_base=None):
    """Simula N cajeros M/M/1 independientes. Retorna dict {nombre: [clientes]}."""
    resultados = {}
    for i in range(num_cajeros):
        semilla = (semilla_base * 97 + i * 13 + 7) if semilla_base is not None else None
        resultados[f'Cajero {i+1}'] = simular_cajero_mm1(tiempo_sim, semilla)
    return resultados


def calcular_estadisticas(clientes):
    """Devuelve dict con métricas de rendimiento de una lista de clientes."""
    if not clientes:
        return {k: 0 for k in [
            'num_clientes', 'espera_promedio', 'espera_max', 'servicio_promedio',
            'tiempo_sistema_promedio', 'tiempo_sistema_max', 'utilizacion',
            'espera_std', 'servicio_std', 'en_sistema_std', 'tipos_count']}

    esperas    = [c['espera']     for c in clientes]
    servicios  = [c['servicio']   for c in clientes]
    en_sistema = [c['en_sistema'] for c in clientes]

    tipos_count = defaultdict(int)
    for c in clientes:
        tipos_count[c['tipo']] += 1

    total_servicio = sum(servicios)
    tiempo_total   = clientes[-1]['salida']

    return {
        'num_clientes':            len(clientes),
        'espera_promedio':         np.mean(esperas),
        'espera_max':              np.max(esperas),
        'servicio_promedio':       np.mean(servicios),
        'tiempo_sistema_promedio': np.mean(en_sistema),
        'tiempo_sistema_max':      np.max(en_sistema),
        'utilizacion':             min(total_servicio / max(tiempo_total, 1), 1.0),
        'espera_std':              np.std(esperas),
        'servicio_std':            np.std(servicios),
        'en_sistema_std':          np.std(en_sistema),
        'tipos_count':             dict(tipos_count),
    }


# --- 1. DETERMINACION DEL ESTADO ESTABLE ---

def analisis_estado_estable():
    """
    Usa la técnica de la media acumulada sobre múltiples réplicas para
    identificar cuántas réplicas se necesitan para alcanzar el estado estable.
    """
    print("=" * 65)
    print("  1. DETERMINACIÓN DEL ESTADO ESTABLE")
    print("=" * 65)

    promedios_replica = []
    esperas_replica   = []

    for r in range(NUM_REPLICAS):
        resultado = simular_sistema(NUM_CAJEROS, TIEMPO_SIMULACION, semilla_base=r)
        ts, te = [], []
        for clientes in resultado.values():
            ts.extend(c['en_sistema'] for c in clientes)
            te.extend(c['espera']     for c in clientes)
        if ts:
            promedios_replica.append(np.mean(ts))
            esperas_replica.append(np.mean(te))

    promedios_replica = np.array(promedios_replica)
    esperas_replica   = np.array(esperas_replica)

    # Media acumulada
    media_acum        = np.cumsum(promedios_replica) / np.arange(1, len(promedios_replica)+1)
    media_espera_acum = np.cumsum(esperas_replica)   / np.arange(1, len(esperas_replica)+1)

    # Criterio: variación relativa < 1 % sostenida durante 5 réplicas
    punto_estable = 10
    for i in range(10, len(media_acum)):
        var_rel = abs(media_acum[i] - media_acum[i-1]) / max(media_acum[i-1], 1e-6)
        if var_rel < 0.01:
            estable = all(
                abs(media_acum[j] - media_acum[j-1]) / max(media_acum[j-1], 1e-6) < 0.02
                for j in range(i+1, min(i+6, len(media_acum)))
            )
            if estable:
                punto_estable = i
                break

    print(f"  Réplicas ejecutadas       : {NUM_REPLICAS}")
    print(f"  Punto de estabilización   : réplica {punto_estable}")
    print(f"  Media estable (en sistema): {media_acum[-1]:.2f} min")
    print(f"  Media estable (espera)    : {media_espera_acum[-1]:.2f} min")

    # ── Gráfica 1: Convergencia ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1a  Tiempo en sistema
    ax = axes[0]
    replicas = np.arange(1, NUM_REPLICAS+1)
    ax.plot(replicas, promedios_replica, 'o', color='#bdc3c7', ms=4, alpha=.5,
            label='Valor por réplica')
    ax.plot(replicas, media_acum, '-', color='#e74c3c', lw=2.5,
            label='Media acumulada')
    ax.axvline(punto_estable, color='#2ecc71', ls='--', lw=2,
               label=f'Estado estable (réplica {punto_estable})')
    ax.axhline(media_acum[-1], color='#3498db', ls=':', alpha=.7,
               label=f'Media final = {media_acum[-1]:.2f} min')
    # banda ±1 std acumulada
    stds = [np.std(promedios_replica[:i+1]) for i in range(len(promedios_replica))]
    ax.fill_between(replicas, media_acum - stds, media_acum + stds,
                    alpha=.08, color='#e74c3c')
    ax.set(xlabel='Número de réplica', ylabel='Tiempo promedio en sistema (min)',
           title='Convergencia – Tiempo en sistema')
    ax.legend(loc='upper right', fontsize=8)

    # 1b  Espera
    ax = axes[1]
    ax.plot(replicas, esperas_replica, 'o', color='#bdc3c7', ms=4, alpha=.5,
            label='Valor por réplica')
    ax.plot(replicas, media_espera_acum, '-', color='#9b59b6', lw=2.5,
            label='Media acumulada')
    ax.axvline(punto_estable, color='#2ecc71', ls='--', lw=2,
               label=f'Estado estable (réplica {punto_estable})')
    ax.axhline(media_espera_acum[-1], color='#3498db', ls=':', alpha=.7,
               label=f'Media final = {media_espera_acum[-1]:.2f} min')
    ax.set(xlabel='Número de réplica', ylabel='Tiempo promedio de espera (min)',
           title='Convergencia – Tiempo de espera')
    ax.legend(loc='upper right', fontsize=8)

    fig.suptitle('Determinación del Estado Estable – Técnica del Promedio',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURAS_DIR, '01_estado_estable.png'))
    plt.close(fig)

    return promedios_replica, media_acum, punto_estable


# --- 2. ESTADISTICAS POR CAJERO ---

def estadisticas_cajeros(num_replicas_estable):
    """Calcula y grafica estadísticas de cada cajero con las réplicas del estado estable."""
    print("\n" + "=" * 65)
    print("  2. ESTADÍSTICAS POR CAJERO")
    print("=" * 65)

    acum  = defaultdict(lambda: defaultdict(list))
    tipos = defaultdict(lambda: defaultdict(list))

    for r in range(num_replicas_estable):
        resultado = simular_sistema(NUM_CAJEROS, TIEMPO_SIMULACION, semilla_base=r+100)
        for cajero, clientes in resultado.items():
            st = calcular_estadisticas(clientes)
            for k, v in st.items():
                if k == 'tipos_count':
                    for tp, cnt in v.items():
                        tipos[cajero][tp].append(cnt)
                elif isinstance(v, (int, float)):
                    acum[cajero][k].append(v)

    # Promediar
    stats = {c: {k: np.mean(v) for k, v in acum[c].items()} for c in acum}
    tipos_prom = {}
    for c in tipos:
        tipos_prom[c] = {tp: np.mean(tipos[c].get(tp, [0])) for tp in TIPOS_USUARIO}

    # Tabla consola
    cajeros = sorted(stats.keys())
    metricas_tabla = [
        ('Clientes atendidos',        'num_clientes',            '{:.0f}'),
        ('Espera promedio (min)',      'espera_promedio',         '{:.2f}'),
        ('Espera máxima (min)',        'espera_max',              '{:.2f}'),
        ('Servicio promedio (min)',    'servicio_promedio',       '{:.2f}'),
        ('Tiempo en sistema (min)',    'tiempo_sistema_promedio', '{:.2f}'),
        ('Utilización (%)',            'utilizacion',             '{:.1f}%'),
    ]
    header = f"  {'Métrica':<30}"
    for c in cajeros:
        header += f" | {c:>12}"
    print(f"\n{header}")
    print("  " + "-" * (30 + 15*len(cajeros)))
    for nombre, key, fmt in metricas_tabla:
        row = f"  {nombre:<30}"
        for c in cajeros:
            v = stats[c].get(key, 0)
            if key == 'utilizacion':
                row += f" | {v*100:>11.1f}%"
            else:
                row += f" | {v:>12.2f}"
        print(row)

    # Cajeros extremos
    svc = {c: stats[c]['servicio_promedio'] for c in cajeros}
    menor = min(svc, key=svc.get)
    mayor = max(svc, key=svc.get)
    print(f"\n  ► Cajero con MENOR tiempo promedio de atención: {menor} ({svc[menor]:.2f} min)")
    print(f"  ► Cajero con MAYOR tiempo promedio de atención: {mayor} ({svc[mayor]:.2f} min)")

    # Totales por tipo
    total_tipo = defaultdict(float)
    for c in tipos_prom:
        for tp, cnt in tipos_prom[c].items():
            total_tipo[tp] += cnt

    print("\n  Promedio de usuarios por tipo (totalidad de cajeros):")
    for tp in TIPOS_USUARIO:
        print(f"    {tp:<12}: {total_tipo[tp]:>7.1f} usuarios")

    # ── Gráfica 2: Estadísticas por cajero (4 subplots) ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    x = np.arange(len(cajeros))
    w = 0.45

    # 2a Servicio promedio
    ax = axes[0, 0]
    vals = [stats[c]['servicio_promedio'] for c in cajeros]
    bars = ax.bar(x, vals, w, color=[COLORES_CAJERO[i] for i in range(len(cajeros))],
                  edgecolor='white', lw=1.5)
    ax.set(xticks=x, xticklabels=cajeros, ylabel='Minutos',
           title='Tiempo Promedio de Atención (Servicio)')
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+.05,
                f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

    # 2b Espera vs Sistema
    ax = axes[0, 1]
    ve = [stats[c]['espera_promedio'] for c in cajeros]
    vs = [stats[c]['tiempo_sistema_promedio'] for c in cajeros]
    ax.bar(x-.18, ve, .34, label='Espera', color='#e74c3c', alpha=.85, edgecolor='white')
    ax.bar(x+.18, vs, .34, label='En sistema', color='#3498db', alpha=.85, edgecolor='white')
    ax.set(xticks=x, xticklabels=cajeros, ylabel='Minutos',
           title='Espera vs Tiempo en Sistema')
    ax.legend()

    # 2c Utilización
    ax = axes[1, 0]
    vu = [stats[c]['utilizacion']*100 for c in cajeros]
    bars = ax.bar(x, vu, w, color=[COLORES_CAJERO[i] for i in range(len(cajeros))],
                  edgecolor='white', lw=1.5)
    ax.axhline(100, color='red',    ls='--', alpha=.5, label='Capacidad 100 %')
    ax.axhline(80,  color='orange', ls='--', alpha=.6, label='Umbral 80 %')
    ax.set(xticks=x, xticklabels=cajeros, ylabel='%',
           title='Utilización por Cajero', ylim=(0, 115))
    ax.legend()
    for b, v in zip(bars, vu):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 2d Clientes atendidos
    ax = axes[1, 1]
    vc = [stats[c]['num_clientes'] for c in cajeros]
    bars = ax.bar(x, vc, w, color=[COLORES_CAJERO[i] for i in range(len(cajeros))],
                  edgecolor='white', lw=1.5)
    ax.set(xticks=x, xticklabels=cajeros, ylabel='Clientes',
           title='Clientes Atendidos por Cajero')
    for b, v in zip(bars, vc):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+.5,
                f'{v:.0f}', ha='center', va='bottom', fontweight='bold')

    fig.suptitle('Análisis Estadístico por Cajero', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURAS_DIR, '02_estadisticas_cajeros.png'))
    plt.close(fig)

    # ── Gráfica 3: Distribución de tipos ──
    fig, ax = plt.subplots(figsize=(12, 7))
    bottom = np.zeros(len(cajeros))
    for tp in TIPOS_USUARIO:
        vals = [tipos_prom[c].get(tp, 0) for c in cajeros]
        ax.bar(cajeros, vals, bottom=bottom, label=tp,
               color=COLORES_TIPO[tp], edgecolor='white', lw=1)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 2:
                ax.text(i, b+v/2, f'{v:.1f}', ha='center', va='center',
                        fontweight='bold', fontsize=9, color='white')
        bottom += vals
    ax.set(ylabel='Número promedio de usuarios',
           title='Distribución de Tipos de Usuario por Cajero')
    ax.legend(title='Tipo de usuario', loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURAS_DIR, '03_distribucion_tipos.png'))
    plt.close(fig)

    return stats, tipos_prom, dict(total_tipo)


# --- 3. ANALISIS DE ESCENARIOS ---

def analisis_escenarios():
    """Compara rendimiento con 3, 4, 5 y 6 cajeros."""
    print("\n" + "=" * 65)
    print("  3. ANÁLISIS DE ESCENARIOS – ESTRATEGIA DE MEJORA")
    print("=" * 65)

    escenarios = [3, 4, 5, 6]
    res_esc = {}

    for nc in escenarios:
        esperas, sistemas, utils = [], [], []
        for r in range(30):
            resultado = simular_sistema(nc, TIEMPO_SIMULACION, semilla_base=r+200)
            for clientes in resultado.values():
                esperas.extend(c['espera'] for c in clientes)
                sistemas.extend(c['en_sistema'] for c in clientes)
                if clientes:
                    t_serv = sum(c['servicio'] for c in clientes)
                    utils.append(t_serv / max(clientes[-1]['salida'], 1))

        res_esc[nc] = {
            'espera_promedio':  np.mean(esperas),
            'sistema_promedio': np.mean(sistemas),
            'espera_p95':       np.percentile(esperas, 95),
            'utilizacion':      np.mean(utils) * 100,
        }
        print(f"\n  {nc} cajeros:")
        for k, v in res_esc[nc].items():
            label = k.replace('_', ' ').title()
            print(f"    {label:<25}: {v:.2f}" + (' %' if 'util' in k else ' min'))

    # ── Gráfica 4: Comparación escenarios ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    x  = np.arange(len(escenarios))
    ce = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']

    for idx, (key, ylabel, titulo) in enumerate([
        ('espera_promedio',  'Minutos', 'Tiempo Promedio de Espera'),
        ('sistema_promedio', 'Minutos', 'Tiempo Promedio en Sistema'),
        ('utilizacion',     '%',       'Utilización Promedio'),
        ('espera_p95',      'Minutos', 'Percentil 95 de Espera'),
    ]):
        ax = axes[idx // 2, idx % 2]
        vals = [res_esc[n][key] for n in escenarios]
        bars = ax.bar(x, vals, .5, color=ce, edgecolor='white', lw=1.5)
        ax.set(xticks=x, xticklabels=[str(n) for n in escenarios],
               xlabel='Número de cajeros', ylabel=ylabel, title=titulo)
        if 'util' in key:
            ax.axhline(80, color='orange', ls='--', alpha=.6, label='Umbral 80 %')
            ax.set_ylim(0, 115)
            ax.legend()
        for b, v in zip(bars, vals):
            fmt = f'{v:.1f}%' if 'util' in key else f'{v:.2f}'
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+.2,
                    fmt, ha='center', va='bottom', fontweight='bold', fontsize=9)

    fig.suptitle('Análisis de Escenarios: Impacto del Número de Cajeros',
                 fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURAS_DIR, '04_analisis_escenarios.png'))
    plt.close(fig)

    return res_esc


# --- 4. VERIFICACION, CALIBRACION Y VALIDACION ---

def verificacion_calibracion_validacion():
    """Compara resultados simulados vs. teóricos M/M/1 y genera diagrama V&V."""
    print("\n" + "=" * 65)
    print("  4. VERIFICACIÓN, CALIBRACIÓN Y VALIDACIÓN")
    print("=" * 65)

    # Parámetros efectivos
    E_interarr = sum(p * MEDIA_LLEGADA[t]  for p, t in zip(PROBABILIDADES, TIPOS_USUARIO))
    E_servicio = sum(p * MEDIA_SERVICIO[t] for p, t in zip(PROBABILIDADES, TIPOS_USUARIO))
    lam = 1.0 / E_interarr          # tasa efectiva de llegada
    mu  = 1.0 / E_servicio           # tasa efectiva de servicio
    rho = lam / mu                   # intensidad de tráfico

    # Fórmulas teóricas M/M/1
    Wq_t = rho / (mu * (1 - rho))   # espera en cola
    W_t  = 1.0 / (mu - lam)         # tiempo en sistema
    Lq_t = rho**2 / (1 - rho)       # longitud de cola
    L_t  = rho / (1 - rho)          # clientes en sistema

    print(f"\n  Parámetros efectivos:")
    print(f"    E[inter-llegada] = {E_interarr:.2f} min  →  λ = {lam:.4f} cl/min")
    print(f"    E[servicio]      = {E_servicio:.2f} min  →  μ = {mu:.4f} cl/min")
    print(f"    ρ (teórico)      = {rho:.4f}")
    print(f"\n  Valores teóricos M/M/1:")
    print(f"    Wq = {Wq_t:.2f} min | W = {W_t:.2f} min | Lq = {Lq_t:.2f} | L = {L_t:.2f}")

    # Valores simulados (30 réplicas, cajero individual)
    Wq_sims, W_sims, rho_sims = [], [], []
    for r in range(30):
        cl = simular_cajero_mm1(TIEMPO_SIMULACION, semilla=r+300)
        if cl:
            Wq_sims.append(np.mean([c['espera']     for c in cl]))
            W_sims.append(np.mean([c['en_sistema'] for c in cl]))
            rho_sims.append(sum(c['servicio'] for c in cl) / max(cl[-1]['salida'], 1))

    Wq_s  = np.mean(Wq_sims)
    W_s   = np.mean(W_sims)
    rho_s = np.mean(rho_sims)

    pct = lambda sim, teo: abs(sim-teo)/max(abs(teo), 1e-6)*100

    print(f"\n  Valores simulados (30 réplicas):")
    print(f"    Wq  = {Wq_s:.2f}  (error {pct(Wq_s, Wq_t):.1f} %)")
    print(f"    W   = {W_s:.2f}  (error {pct(W_s, W_t):.1f} %)")
    print(f"    ρ   = {rho_s:.4f}  (error {pct(rho_s, rho):.1f} %)")

    # ── Gráfica 5: Teórico vs Simulado + Boxplot ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    labels = ['ρ (utilización)', 'Wq (espera)', 'W (en sistema)']
    teo = [rho, Wq_t, W_t]
    sim = [rho_s, Wq_s, W_s]
    xv = np.arange(len(labels))
    ax.bar(xv-.18, teo, .34, label='Teórico M/M/1', color='#3498db', edgecolor='white')
    ax.bar(xv+.18, sim, .34, label='Simulación',     color='#e74c3c', edgecolor='white')
    ax.set(xticks=xv, xticklabels=labels, ylabel='Valor',
           title='Verificación: Teórico vs. Simulado')
    ax.legend()
    # errores
    for i, (t, s) in enumerate(zip(teo, sim)):
        ax.text(i, max(t, s)+.3, f'error {pct(s, t):.1f}%',
                ha='center', fontsize=8, color='gray')

    ax = axes[1]
    bp = ax.boxplot([rho_sims, Wq_sims, W_sims], labels=['ρ', 'Wq', 'W'],
                    patch_artist=True,
                    boxprops=dict(facecolor='#3498db', alpha=.7),
                    medianprops=dict(color='#e74c3c', lw=2))
    ax.set(title='Validación: Variabilidad entre Réplicas', ylabel='Valor')

    fig.suptitle('Verificación, Calibración y Validación', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURAS_DIR, '05_verificacion_validacion.png'))
    plt.close(fig)

    # ── Gráfica 6: Diagrama de flujo del proceso V&V ──
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')

    def draw_box(x, y, titulo, color, detalle, w=2.4, h=1.6):
        rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='white',
                              linewidth=2, alpha=.88, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y+.3, titulo, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', zorder=4)
        ax.text(x, y-.3, detalle, ha='center', va='center',
                fontsize=7, color='white', alpha=.9, zorder=4,
                linespacing=1.4)

    bloques = [
        (2,   9.5, 'MODELO\nCONCEPTUAL',    '#3498db',
         'M/M/1 con 4 tipos\nde usuario'),
        (5,   9.5, 'VERIFICACIÓN',            '#2ecc71',
         'Comparar simulación\nvs fórmulas teóricas'),
        (8,   9.5, 'MODELO\nCOMPUTACIONAL',  '#9b59b6',
         'Python – eventos\ndiscretos'),
        (2,   6.5, 'CALIBRACIÓN',             '#f39c12',
         f'Ajustar λ,μ\nρ_sim≈{rho_s:.3f} ≈ ρ_teo≈{rho:.3f}'),
        (5,   6.5, 'VALIDACIÓN',              '#e74c3c',
         'Sensibilidad\nMúltiples semillas\nIC 95 %'),
        (8,   6.5, 'SISTEMA\nREAL',           '#1abc9c',
         'Parqueadero\nSupercentro'),
        (5,   3.5, 'RESULTADOS\nFINALES',     '#2c3e50',
         f'ρ={rho_s:.3f} Wq={Wq_s:.1f}\nW={W_s:.1f} min'),
    ]
    for b in bloques:
        draw_box(*b)

    flechas = [
        (3.2, 9.5, 3.8, 9.5), (6.2, 9.5, 6.8, 9.5),
        (2,   8.7, 2,   7.3), (5,   8.7, 5,   7.3),
        (8,   8.7, 8,   7.3),
        (3.2, 6.5, 3.8, 6.5), (6.2, 6.5, 6.8, 6.5),
        (5,   5.7, 5,   4.3),
    ]
    for x1, y1, x2, y2 in flechas:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='white', lw=2), zorder=5)

    ax.set_title('Diagrama del Proceso de Verificación, Calibración y Validación',
                 fontsize=14, fontweight='bold', color='white', pad=20)
    fig.patch.set_facecolor('#1a1a2e')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURAS_DIR, '06_diagrama_vv.png'), facecolor='#1a1a2e')
    plt.close(fig)

    return {
        'lambda': lam, 'mu': mu,
        'rho_teorico': rho, 'Wq_teorico': Wq_t, 'W_teorico': W_t,
        'Lq_teorico': Lq_t, 'L_teorico': L_t,
        'rho_sim': rho_s, 'Wq_sim': Wq_s, 'W_sim': W_s,
    }


# --- 5. ELIMINACION DEL ESTADO TRANSITORIO ---

def eliminacion_transitorio():
    """Identifica y elimina el estado transitorio. Muestra antes y después."""
    print("\n" + "=" * 65)
    print("  5. ELIMINACIÓN DEL ESTADO TRANSITORIO")
    print("=" * 65)

    # Simulación larga para visualizar transitorio
    clientes = simular_cajero_mm1(TIEMPO_SIMULACION, semilla=42)
    if not clientes:
        print("  Sin clientes generados.")
        return {}

    ts = np.array([c['en_sistema'] for c in clientes])
    n  = len(ts)
    media_acum = np.cumsum(ts) / np.arange(1, n+1)

    # Método de Welch: promediar sobre réplicas
    NR_W, MAX_OBS = 20, 200
    series = []
    for r in range(NR_W):
        cl = simular_cajero_mm1(TIEMPO_SIMULACION, semilla=r+500)
        series.append([c['en_sistema'] for c in cl[:MAX_OBS]])
    min_len = min(len(s) for s in series)
    welch = np.mean([s[:min_len] for s in series], axis=0)

    # Suavizar
    win = max(1, min(10, min_len//5))
    welch_s = np.convolve(welch, np.ones(win)/win, mode='valid')

    # Punto de truncamiento
    media_tail = np.mean(welch_s[-max(1, len(welch_s)//2):])
    trunc = 0
    for i in range(len(welch_s)):
        if abs(welch_s[i] - media_tail) / max(media_tail, 1e-6) < 0.15:
            trunc = i
            break
    trunc = max(5, min(trunc, n//3))

    m_antes  = np.mean(ts)
    m_desp   = np.mean(ts[trunc:])
    s_antes  = np.std(ts)
    s_desp   = np.std(ts[trunc:])

    print(f"  Observaciones totales        : {n}")
    print(f"  Punto de truncamiento        : observación {trunc}")
    print(f"  ANTES → Media: {m_antes:.2f} min  Std: {s_antes:.2f}")
    print(f"  DESPUÉS → Media: {m_desp:.2f} min  Std: {s_desp:.2f}")

    # ── Gráfica 7: Antes / Después ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 7a  Serie completa
    ax = axes[0, 0]
    ax.plot(ts, 'o', color='#bdc3c7', ms=3, alpha=.3, label='Observaciones')
    ax.plot(media_acum, '-', color='#e74c3c', lw=2, label='Media acumulada')
    ax.axvline(trunc, color='#2ecc71', ls='--', lw=2,
               label=f'Truncamiento ({trunc})')
    ax.axvspan(0, trunc, alpha=.08, color='red')
    ax.set(xlabel='Observación', ylabel='Tiempo en sistema (min)',
           title='ANTES: Serie con estado transitorio')
    ax.legend(fontsize=8)

    # 7b  Solo estable
    ax = axes[0, 1]
    est = ts[trunc:]
    ma_est = np.cumsum(est) / np.arange(1, len(est)+1)
    ax.plot(est, 'o', color='#bdc3c7', ms=3, alpha=.3, label='Observaciones')
    ax.plot(ma_est, '-', color='#2ecc71', lw=2, label='Media acumulada')
    ax.axhline(m_desp, color='#3498db', ls=':', lw=2,
               label=f'Media = {m_desp:.2f}')
    ax.set(xlabel='Observación (desde truncamiento)',
           ylabel='Tiempo en sistema (min)',
           title='DESPUÉS: Solo estado estable')
    ax.legend(fontsize=8)

    # 7c  Welch
    ax = axes[1, 0]
    ax.plot(welch[:min_len], '-', color='#9b59b6', alpha=.5, lw=1, label='Welch crudo')
    ax.plot(welch_s, '-', color='#e74c3c', lw=2.5, label='Welch suavizado')
    ax.axvline(trunc, color='#2ecc71', ls='--', lw=2, label=f'Truncamiento ({trunc})')
    ax.set(xlabel='Observación', ylabel='Promedio de Welch',
           title='Método de Welch – Identificación del transitorio')
    ax.legend(fontsize=8)

    # 7d  Histogramas
    ax = axes[1, 1]
    ax.hist(ts, bins=25, alpha=.5, color='#e74c3c', density=True,
            edgecolor='white', label=f'Antes (μ={m_antes:.2f})')
    ax.hist(est, bins=25, alpha=.5, color='#2ecc71', density=True,
            edgecolor='white', label=f'Después (μ={m_desp:.2f})')
    ax.axvline(m_antes, color='#e74c3c', ls='--', lw=2)
    ax.axvline(m_desp,  color='#2ecc71', ls='--', lw=2)
    ax.set(xlabel='Tiempo en sistema (min)', ylabel='Densidad',
           title='Distribución: Antes vs. Después')
    ax.legend(fontsize=9)

    fig.suptitle('Eliminación del Estado Transitorio – Antes y Después',
                 fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURAS_DIR, '07_estado_transitorio.png'))
    plt.close(fig)

    return {
        'punto_truncamiento': trunc,
        'media_antes': m_antes, 'media_despues': m_desp,
        'std_antes': s_antes,   'std_despues': s_desp,
    }


# --- 6. GRAFICA RESUMEN ---

def grafica_resumen(stats, res_esc, vv):
    """Panel resumen de 6 subplots."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    cajeros = sorted(stats.keys())

    # 1 Utilización
    ax = axes[0, 0]
    u = [stats[c]['utilizacion']*100 for c in cajeros]
    bars = ax.bar(cajeros, u, color=[COLORES_CAJERO[i] for i in range(len(cajeros))],
                  edgecolor='white')
    ax.axhline(80, color='orange', ls='--', alpha=.6)
    ax.set(title='Utilización', ylabel='%', ylim=(0, 115))
    for b, v in zip(bars, u):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

    # 2 Servicio
    ax = axes[0, 1]
    s = [stats[c]['servicio_promedio'] for c in cajeros]
    ax.bar(cajeros, s, color=[COLORES_CAJERO[i] for i in range(len(cajeros))],
           edgecolor='white')
    ax.set(title='Servicio Promedio', ylabel='min')

    # 3 Espera
    ax = axes[0, 2]
    e = [stats[c]['espera_promedio'] for c in cajeros]
    ax.bar(cajeros, e, color=[COLORES_CAJERO[i] for i in range(len(cajeros))],
           edgecolor='white')
    ax.set(title='Espera Promedio', ylabel='min')

    # 4 Espera vs Nº cajeros
    ax = axes[1, 0]
    ks = sorted(res_esc)
    ax.plot(ks, [res_esc[n]['espera_promedio'] for n in ks],
            'o-', color='#e74c3c', lw=2.5, ms=10)
    ax.set(title='Espera vs Nº Cajeros', xlabel='Cajeros', ylabel='min')
    ax.set_xticks(ks)

    # 5 Teórico vs Sim
    ax = axes[1, 1]
    lb = ['ρ', 'Wq', 'W']
    t = [vv['rho_teorico'], vv['Wq_teorico'], vv['W_teorico']]
    s = [vv['rho_sim'],     vv['Wq_sim'],     vv['W_sim']]
    xv = np.arange(3)
    ax.bar(xv-.18, t, .34, label='Teórico', color='#3498db', edgecolor='white')
    ax.bar(xv+.18, s, .34, label='Simulado', color='#e74c3c', edgecolor='white')
    ax.set(xticks=xv, xticklabels=lb, title='Teórico vs. Simulado')
    ax.legend()

    # 6 Distribución tipos
    ax = axes[1, 2]
    ax.pie(PROBABILIDADES, labels=TIPOS_USUARIO,
           colors=[COLORES_TIPO[t] for t in TIPOS_USUARIO],
           autopct='%1.1f%%', startangle=90,
           textprops={'fontsize': 10})
    ax.set_title('Distribución de Usuarios')

    fig.suptitle('Resumen General – Simulación Parqueadero Supercentro',
                 fontsize=16, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURAS_DIR, '08_resumen_general.png'))
    plt.close(fig)


# --- MAIN ---

def main():
    print()
    print("=" * 60)
    print("  SIMULACION DE PARQUEADEROS - CENTRO COMERCIAL SUPERCENTRO")
    print("  Modelo M/M/1 - Simulacion por Eventos Discretos")
    print("  Alejandro Arango Calderon")
    print("=" * 60)
    print()

    # 1. Estado estable
    _, media_acum, punto_estable = analisis_estado_estable()
    n_rep = max(punto_estable + 5, 25)

    # 2. Estadisticas
    stats, tipos_prom, total_tipo = estadisticas_cajeros(n_rep)

    # 3. Escenarios
    res_esc = analisis_escenarios()

    # 4. V&V
    vv = verificacion_calibracion_validacion()

    # 5. Transitorio
    trans = eliminacion_transitorio()

    # 6. Resumen
    grafica_resumen(stats, res_esc, vv)

    # Resumen final
    print("\n" + "=" * 65)
    print("  RESUMEN Y RECOMENDACION FINAL")
    print("=" * 65)

    svc = {c: stats[c]['servicio_promedio'] for c in stats}
    esp = {c: stats[c]['espera_promedio']   for c in stats}
    print(f"\n  Estado estable alcanzado en réplica {punto_estable}; usadas {n_rep}")
    print(f"  Cajero menor servicio : {min(svc, key=svc.get)} ({svc[min(svc, key=svc.get)]:.2f} min)")
    print(f"  Cajero mayor servicio : {max(svc, key=svc.get)} ({svc[max(svc, key=svc.get)]:.2f} min)")
    print(f"  Cajero menor espera   : {min(esp, key=esp.get)} ({esp[min(esp, key=esp.get)]:.2f} min)")
    print(f"  Cajero mayor espera   : {max(esp, key=esp.get)} ({esp[max(esp, key=esp.get)]:.2f} min)")

    print(f"\n  Usuarios promedio por tipo (3 cajeros):")
    for tp in TIPOS_USUARIO:
        print(f"    {tp:<12}: {total_tipo[tp]:.1f}")

    rho3 = res_esc[3]['utilizacion']
    rho4 = res_esc[4]['utilizacion']
    print(f"\n  CRITERIO DE DECISIÓN:")
    print(f"    Utilización recomendada < 80 %")
    print(f"    Espera P95 < 15 min")
    if rho3 > 80:
        print(f"\n  ⚠  Con 3 cajeros ρ = {rho3:.1f} % > 80 %  → AGREGAR cajeros")
        print(f"     Con 4 cajeros ρ = {rho4:.1f} %")
        print(f"     Recomendación: instalar al menos 4 cajeros por salida.")
    else:
        print(f"\n  ✓  Con 3 cajeros ρ = {rho3:.1f} % ≤ 80 %  → SUFICIENTE")
        print(f"     Se recomienda mantener 3 cajeros y monitorear en horas pico.")
        if rho3 > 70:
            print(f"     NOTA: la utilización está cercana al umbral; considerar")
            print(f"     un 4.º cajero para absorber picos de demanda.")

    # Guardar resultados en TXT
    ruta_txt = os.path.join(OUTPUT_DIR, 'resultados_simulacion.txt')
    with open(ruta_txt, 'w', encoding='utf-8') as f:
        f.write("RESULTADOS DE LA SIMULACIÓN DE PARQUEADEROS\n")
        f.write("Centro Comercial Supercentro\n")
        f.write("Autor: Alejandro Arango Calderón\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Estado estable en réplica {punto_estable}, réplicas usadas: {n_rep}\n\n")

        f.write("ESTADÍSTICAS POR CAJERO\n" + "-"*40 + "\n")
        for c in sorted(stats):
            s = stats[c]
            f.write(f"\n{c}:\n")
            f.write(f"  Clientes     : {s['num_clientes']:.0f}\n")
            f.write(f"  Espera prom  : {s['espera_promedio']:.2f} min\n")
            f.write(f"  Servicio prom: {s['servicio_promedio']:.2f} min\n")
            f.write(f"  En sistema   : {s['tiempo_sistema_promedio']:.2f} min\n")
            f.write(f"  Utilización  : {s['utilizacion']*100:.1f} %\n")

        f.write(f"\nMenor servicio: {min(svc, key=svc.get)} ({svc[min(svc, key=svc.get)]:.2f} min)\n")
        f.write(f"Mayor servicio: {max(svc, key=svc.get)} ({svc[max(svc, key=svc.get)]:.2f} min)\n")

        f.write("\n\nUSUARIOS POR TIPO\n" + "-"*40 + "\n")
        for tp in TIPOS_USUARIO:
            f.write(f"  {tp:<12}: {total_tipo[tp]:.1f}\n")

        f.write("\n\nESCENARIOS\n" + "-"*40 + "\n")
        for n in sorted(res_esc):
            r = res_esc[n]
            f.write(f"\n{n} cajeros: espera={r['espera_promedio']:.2f} min, "
                    f"sistema={r['sistema_promedio']:.2f} min, util={r['utilizacion']:.1f}%\n")

        f.write("\n\nVERIFICACIÓN\n" + "-"*40 + "\n")
        f.write(f"  ρ  teórico={vv['rho_teorico']:.4f}  simulado={vv['rho_sim']:.4f}\n")
        f.write(f"  Wq teórico={vv['Wq_teorico']:.2f}  simulado={vv['Wq_sim']:.2f}\n")
        f.write(f"  W  teórico={vv['W_teorico']:.2f}   simulado={vv['W_sim']:.2f}\n")

        f.write("\n\nESTADO TRANSITORIO\n" + "-"*40 + "\n")
        f.write(f"  Truncamiento: obs {trans['punto_truncamiento']}\n")
        f.write(f"  Media antes : {trans['media_antes']:.2f} min\n")
        f.write(f"  Media después: {trans['media_despues']:.2f} min\n")

    print(f"\n  Gráficas guardadas en: {FIGURAS_DIR}")
    print(f"  Resultados en: {ruta_txt}")
    print("\n  ✅ Simulación completada exitosamente.\n")


if __name__ == '__main__':
    main()
