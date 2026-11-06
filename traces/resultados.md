# Resultados de los escenarios

| # | Escenario | Modo | Ruta del supervisor | Decisiones | Refin. | Validación | ¿Entra? | Datos (memoria · web) | Búsquedas web | Traza |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | flujo completo | real | sup → inv → sup → inv → sup → ana → sup → síntesis | 4 | 1 | aprobada | sí | 12 · 20 | 8 | [1_flujo_completo_real_20261105-212519.json](traces/1_flujo_completo_real_20261105-212519.json) |
| 1 | flujo completo | mock | sup → inv → sup → ana → sup → síntesis | 3 | 0 | aprobada | sí | 0 · 6 | 5 (simuladas) | [1_flujo_completo_mock_20261105-213218.json](traces/1_flujo_completo_mock_20261105-213218.json) |
| 2 | solo clima | real | sup → inv → sup → síntesis | 2 | 0 | no se validó | — | 3 · 3 | 1 | [2_solo_clima_real_20261105-213409.json](traces/2_solo_clima_real_20261105-213409.json) |
| 2 | solo clima | mock | sup → inv → sup → síntesis | 2 | 0 | no se validó | — | 1 · 0 | 0 (simuladas) | [2_solo_clima_mock_20261105-213218.json](traces/2_solo_clima_mock_20261105-213218.json) |
| 3 | no alcanza | real | sup → inv → sup → inv → sup → ana → sup → síntesis | 4 | 1 | aprobada | no | 23 · 0 | 0 | [3_no_alcanza_real_20261105-213431.json](traces/3_no_alcanza_real_20261105-213431.json) |
| 3 | no alcanza | mock | sup → inv → sup → ana → sup → síntesis | 3 | 0 | aprobada | no | 6 · 0 | 0 (simuladas) | [3_no_alcanza_mock_20261105-213218.json](traces/3_no_alcanza_mock_20261105-213218.json) |
| 4 | fechas invalidas | real | sup → inv → sup → inv → sup → ana → sup → síntesis | 4 | 1 | rechazada | — | 23 · 6 | 2 | [4_fechas_invalidas_real_20261105-213600.json](traces/4_fechas_invalidas_real_20261105-213600.json) |
| 4 | fechas invalidas | mock | sup → inv → sup → ana → sup → síntesis | 3 | 0 | rechazada | — | 6 · 0 | 0 (simuladas) | [4_fechas_invalidas_mock_20261105-213218.json](traces/4_fechas_invalidas_mock_20261105-213218.json) |
| 5 | memoria | real | sup → inv → sup → ana → sup → inv → sup → síntesis | 4 | 1 | aprobada | no | 22 · 9 | 3 | [5_memoria_real_20261105-213236.json](traces/5_memoria_real_20261105-213236.json) |
| 5 | memoria | mock | sup → inv → sup → ana → sup → síntesis | 3 | 0 | aprobada | sí | 6 · 0 | 0 (simuladas) | [5_memoria_mock_20261105-213218.json](traces/5_memoria_mock_20261105-213218.json) |
| 6 | refinamiento | real | sup → inv → sup → síntesis | 2 | 0 | no se validó | — | 9 · 6 | 2 | [6_refinamiento_real_20261105-213718.json](traces/6_refinamiento_real_20261105-213718.json) |
| 6 | refinamiento | mock | sup → inv → sup → inv → sup → síntesis | 3 | 1 | no se validó | — | 0 · 3 | 2 (simuladas) | [6_refinamiento_mock_20261105-213218.json](traces/6_refinamiento_mock_20261105-213218.json) |
| 7 | tavily falla | mock | sup → inv → sup → inv → sup → ana → sup → síntesis | 4 | 1 | aprobada | — | 0 · 0 | 6 (simuladas) | [7_tavily_falla_mock_20261105-213218.json](traces/7_tavily_falla_mock_20261105-213218.json) |
| 8 | freno de pasos | real | sup → inv → sup → síntesis | 2 (freno) | 0 | no se validó | — | 20 · 3 | 1 | [8_freno_de_pasos_real_20261105-213756.json](traces/8_freno_de_pasos_real_20261105-213756.json) |
| 8 | freno de pasos | mock | sup → inv → sup → síntesis | 2 (freno) | 0 | no se validó | — | 6 · 0 | 0 (simuladas) | [8_freno_de_pasos_mock_20261105-213218.json](traces/8_freno_de_pasos_mock_20261105-213218.json) |
