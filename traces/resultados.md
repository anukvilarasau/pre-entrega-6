# Resultados de los escenarios

| # | Escenario | Modo | Ruta del supervisor | Decisiones | Refin. | Validación | ¿Entra? | Datos (memoria · web) | Búsquedas web | Traza |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | flujo completo | real | sup → inv → sup → ana → sup → síntesis | 3 | 0 | aprobada | no | 0 · 18 | 6 | [1_flujo_completo_real_20261110-150904.json](1_flujo_completo_real_20261110-150904.json) |
| 1 | flujo completo | mock | sup → inv → sup → ana → sup → síntesis | 3 | 0 | aprobada | sí | 0 · 6 | 5 (simuladas) | [1_flujo_completo_mock_20261110-151409.json](1_flujo_completo_mock_20261110-151409.json) |
| 2 | solo clima | real | sup → inv → sup → síntesis | 2 | 0 | no se validó | — | 3 · 0 | 0 | [2_solo_clima_real_20261110-151106.json](2_solo_clima_real_20261110-151106.json) |
| 2 | solo clima | mock | sup → inv → sup → síntesis | 2 | 0 | no se validó | — | 1 · 0 | 0 (simuladas) | [2_solo_clima_mock_20261110-151410.json](2_solo_clima_mock_20261110-151410.json) |
| 3 | no alcanza | real | sup → inv → sup → ana → sup → síntesis | 3 | 0 | aprobada | no | 17 · 0 | 0 | [3_no_alcanza_real_20261110-151120.json](3_no_alcanza_real_20261110-151120.json) |
| 3 | no alcanza | mock | sup → inv → sup → ana → sup → síntesis | 3 | 0 | aprobada | no | 6 · 0 | 0 (simuladas) | [3_no_alcanza_mock_20261110-151410.json](3_no_alcanza_mock_20261110-151410.json) |
| 4 | fechas invalidas | real | sup → inv → sup → ana → sup → síntesis | 3 | 0 | rechazada | — | 15 · 0 | 0 | [4_fechas_invalidas_real_20261110-151218.json](4_fechas_invalidas_real_20261110-151218.json) |
| 4 | fechas invalidas | mock | sup → inv → sup → ana → sup → síntesis | 3 | 0 | rechazada | — | 6 · 0 | 0 (simuladas) | [4_fechas_invalidas_mock_20261110-151410.json](4_fechas_invalidas_mock_20261110-151410.json) |
| 5 | memoria | real | sup → inv → sup → ana → sup → síntesis | 3 | 0 | aprobada | no | 17 · 0 | 0 | [5_memoria_real_20261110-151006.json](5_memoria_real_20261110-151006.json) |
| 5 | memoria | mock | sup → inv → sup → ana → sup → síntesis | 3 | 0 | aprobada | sí | 6 · 0 | 0 (simuladas) | [5_memoria_mock_20261110-151409.json](5_memoria_mock_20261110-151409.json) |
| 6 | refinamiento | real | sup → inv → sup → inv → sup → síntesis | 3 | 1 | no se validó | — | 6 · 0 | 0 | [6_refinamiento_real_20261110-151258.json](6_refinamiento_real_20261110-151258.json) |
| 6 | refinamiento | mock | sup → inv → sup → inv → sup → síntesis | 3 | 1 | no se validó | — | 2 · 1 | 1 (simuladas) | [6_refinamiento_mock_20261110-151410.json](6_refinamiento_mock_20261110-151410.json) |
| 7 | tavily falla | mock | sup → inv → sup → inv → sup → ana → sup → síntesis | 4 | 1 | aprobada | — | 0 · 0 | 6 (simuladas) | [7_tavily_falla_mock_20261110-151410.json](7_tavily_falla_mock_20261110-151410.json) |
| 8 | freno de pasos | real | sup → inv → sup → síntesis | 2 (freno) | 0 | no se validó | — | 17 · 0 | 0 | [8_freno_de_pasos_real_20261110-151330.json](8_freno_de_pasos_real_20261110-151330.json) |
| 8 | freno de pasos | mock | sup → inv → sup → síntesis | 2 (freno) | 0 | no se validó | — | 6 · 0 | 0 (simuladas) | [8_freno_de_pasos_mock_20261110-151410.json](8_freno_de_pasos_mock_20261110-151410.json) |
