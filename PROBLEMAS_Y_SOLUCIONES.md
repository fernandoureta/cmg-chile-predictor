# Problemas y Soluciones — CMG Chile Predictor

Registro cronológico de problemas encontrados durante el desarrollo y cómo se resolvieron.

---

## 2026-04-13 — etl/scrapers/cen_marginal.py

**Problema:** `AmbiguousTimeError` al localizar zona horaria `America/Santiago`. El script fallaba en los 6 archivos TSV con el mensaje `Cannot infer dst time from YYYY-MM-DD 23:00:00, try using the 'ambiguous' argument`.

**Causa raíz:** `ambiguous="first"` no es un valor válido en pandas. Los únicos strings aceptados por `dt.tz_localize()` son `"infer"`, `"NaT"` y `"raise"`. Al pasarle `"first"`, pandas lo ignoraba silenciosamente y caía al comportamiento por defecto `"raise"`, que lanza el error ante cualquier hora ambigua. El intento siguiente con `ambiguous="infer"` también falló porque ese modo necesita ver la hora duplicada en los datos para poder inferir cuál es DST, y esa hora no existe en los archivos del CEN.

**Solución:** Cambiar a `ambiguous=True` (booleano escalar). Este valor le indica a pandas explícitamente que toda hora ambigua debe tratarse como DST=True (UTC-3). Es correcto porque el CEN siempre genera exactamente 24 filas por día (hora 1–24) sin registrar la segunda ocurrencia de la hora repetida del cambio horario. La hora 24 (23:00 local) del día de transición de abril ocurre antes de medianoche, es decir, todavía dentro del periodo DST.

**Lección aprendida:** El CEN no refleja la realidad del reloj físico en los días de cambio horario: simplemente descarta la hora repetida. Cualquier modo de inferencia automática (`"infer"`) fallará porque necesita evidencia en los datos que no existe. Para datasets con cobertura horaria fija de 24h/día, la única solución robusta es declarar explícitamente el offset con un booleano escalar.

---

## 2026-04-13 — etl/scrapers/cen_marginal.py

**Problema:** El filtro de barra comparaba `barra_mnemotecnico == "BA S/E QUILLOTA 220KV BP1-1"` y retornaba un DataFrame vacío, lanzando `ValueError` para todos los archivos.

**Causa raíz:** Confusión entre el código mnemotécnico interno del CEN y el nombre legible de la barra. La columna `barra_mnemotecnico` contiene el código interno (`BA02T0002SE032T0002`), no el nombre descriptivo. El valor `"BA S/E QUILLOTA 220KV BP1-1"` corresponde a otra columna del archivo.

**Solución:** Eliminar el filtro completamente. Los archivos TSV ya vienen filtrados para la barra Quillota desde el formulario de descarga del portal del CEN, por lo que filtrar en código es redundante y propenso a romperse si el CEN cambia su nomenclatura interna.

**Lección aprendida:** Nunca escribir un filtro de datos sin haber inspeccionado el archivo real primero. El nombre de una columna no garantiza nada sobre su contenido. Ante la duda, hacer un `df.head()` o `df.sample()` antes de asumir qué valor buscar.

---

## 2026-04-13 — etl/scrapers/cen_marginal.py

**Problema:** `dayfirst=True` en el parseo de fechas producía conversiones incorrectas. El formato real de los archivos del CEN es ISO `YYYY-MM-DD`, no el formato chileno `DD/MM/YYYY`.

**Causa raíz:** Asunción incorrecta sobre el formato de fecha. Se asumió que, al ser un archivo de una fuente chilena, las fechas seguirían la convención local `DD/MM/YYYY`. El CEN en realidad usa el formato ISO estándar.

**Solución:** Cambiar a `format="%Y-%m-%d"` explícito en `pd.to_datetime()`. El formato explícito es más rápido que la inferencia automática y elimina cualquier ambigüedad (por ejemplo, `01/02/2024` puede ser 1 de febrero o 2 de enero dependiendo del `dayfirst`).

**Lección aprendida:** Nunca asumir el formato de fecha. Siempre inspeccionar el archivo crudo antes de escribir el parser. Preferir siempre `format=` explícito sobre `dayfirst=` o inferencia automática: es más seguro, más rápido y falla de forma clara si el formato cambia.

---

## Próximos problemas registrados

*(se irán agregando a medida que avance el proyecto)*
