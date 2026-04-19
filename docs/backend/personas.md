# Perfiles para DeUna MVP

## Perfil 1: Rosa (dueña de tienda de barrio)
- Contexto: Maneja una tienda pequeña de barrio con apoyo de una vendedora.
- Perfil de alfabetización: Baja alfabetización financiera y técnica.
- Familiaridad con IA: Ninguna.
- Objetivo: Entender si su ingreso diario mejora y en qué días le va peor.
- Preguntas típicas:
  - "¿Cómo me fue esta semana?"
  - "¿Qué día vendí menos?"
  - "¿A qué hora vendo más?"

## Perfil 2: Miguel (dueño de puesto de comida)
- Contexto: Opera con dos vendedores por turnos.
- Perfil de alfabetización: Baja alfabetización técnica y nociones numéricas básicas.
- Familiaridad con IA: Ninguna.
- Objetivo: Comparar desempeño entre vendedores e identificar clientes frecuentes.
- Preguntas típicas:
  - "¿Quién vendió más hoy, yo o mis vendedores?"
  - "¿Cuáles son mis clientes más frecuentes?"
  - "¿Esta semana subí o bajé frente a la pasada?"

## Perfil 3: Daniela (dueña de negocio de belleza)
- Contexto: Trabaja casi siempre sola y controla ingresos de forma manual.
- Perfil de alfabetización: Baja alfabetización financiera.
- Familiaridad con IA: Ninguna.
- Objetivo: Detectar tendencias simples y comportamiento de retorno de clientes.
- Preguntas típicas:
  - "¿Tengo clientes nuevos este mes?"
  - "¿Cuántos clientes regresaron?"
  - "¿Hay clientes que ya no volvieron?"

## Perfil 4: Carlos (dueño de taller pequeño)
- Contexto: Comercio de un solo dueño con demanda irregular.
- Perfil de alfabetización: Baja alfabetización técnica.
- Familiaridad con IA: Ninguna.
- Objetivo: Entender caídas de ingresos y decidir acciones concretas.
- Preguntas típicas:
  - "¿Por qué bajó mi ingreso estos días?"
  - "¿Qué me recomiendas revisar primero?"

## Reglas de estilo de respuesta para todos los perfiles
- Frases cortas y directas.
- Evitar jerga.
- Mencionar números y períodos con claridad.
- Si falta información, decirlo con claridad y sugerir una pregunta cercana que sí se pueda responder.

## Integración con evaluación
- Las preguntas del set de evaluación se configuran en `src/agent/evals/profiles_question_bank.yaml`.
- Cada pregunta se asocia a un `profile_id` para poder regenerar el eval set cuando cambie el diseño de perfiles.
