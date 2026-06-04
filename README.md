# Simulación de Parqueaderos - Centro Comercial Supercentro

Proyecto de simulación por eventos discretos del sistema de pago de parqueaderos del centro comercial Supercentro, usando el modelo de colas M/M/1.

## Descripción

El sistema dispone de 3 cajeros automáticos en cada punto de salida del parqueadero. Los usuarios interactúan con el cajero para registrar la placa de su vehículo y realizar el pago. La velocidad de interacción varía según el tipo de usuario:

| Tipo de Usuario | Tiempo de Servicio (Exp.) | Tiempo entre Llegadas | Proporción |
|:---------------:|:-------------------------:|:---------------------:|:----------:|
| Rápido          | 1 min                     | 3 min                 | 25%        |
| Normal          | 3 min                     | 3 min                 | 20%        |
| Lento           | 4 min                     | 5 min                 | 27.5%      |
| Muy Lento       | 6 min                     | 7 min                 | 27.5%      |

Se busca determinar si 3 cajeros por salida son suficientes o si se necesitan más.

## Estructura

```
├── simulacion_parqueadero.py           # Script principal
├── Arango_Alejandro_problemaParqueadero.md  # Documento del trabajo
├── resultados_simulacion.txt           # Resultados numéricos
└── figuras/                            # Gráficas generadas
    ├── 01_estado_estable.png
    ├── 02_estadisticas_cajeros.png
    ├── 03_distribucion_tipos.png
    ├── 04_analisis_escenarios.png
    ├── 05_verificacion_validacion.png
    ├── 06_diagrama_vv.png
    ├── 07_estado_transitorio.png
    └── 08_resumen_general.png
```

## Requisitos

- Python 3.10 o superior
- NumPy
- Matplotlib

```bash
pip install numpy matplotlib
```

## Cómo ejecutar

```bash
python simulacion_parqueadero.py
```

Se generan automáticamente las 8 gráficas en `figuras/` y los resultados en `resultados_simulacion.txt`.

## Resultados principales

- Estado estable alcanzado en la réplica 19 de 50
- Utilización promedio por cajero: 74% - 77%
- Cajero con menor tiempo de atención: Cajero 2 (3.60 min)
- Cajero con mayor tiempo de atención: Cajero 1 (3.73 min)
- Verificación contra fórmulas teóricas M/M/1 con error < 4%
- Conclusión: 3 cajeros son suficientes para la operación normal

## Autor

Alejandro Arango Calderón
