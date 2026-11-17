# Datathon FME 2026 · Retrospectiva (challenge RawTree)
 
Documento vivo: aquí se va guardando el feedback de Marc sobre su paso por la Datathon FME 2026 (14–15 de noviembre de 2026, UPC Barcelona). Última actualización: 16 de noviembre de 2026.
 
## El reto, en una línea
 
Recomendación en tiempo real (challenge de TinyBird con su base de datos RawTree): para cada petición de un comprador había que devolver 10 productos ordenados en menos de 3 segundos, leyendo y escribiendo en RawTree. La nota automática dependía de la conversión (50), el cold start (10) y la fiabilidad (10); el jurado ponía los otros 30 puntos.
 
## Enlaces
 
- Web del reto: https://datathon.rawtree.dev (novedades en https://datathon.rawtree.dev/#updates)
- Leaderboard final: https://datathon.rawtree.dev/leaderboard
- Página de introducción mostrada en la apertura (fichero para descargar y abrir en el navegador): https://drive.google.com/file/d/1t8RDXXDl62aPnBZZcW-rer1sU-kKSSkA/
- **Simulador publicado tras el evento:** https://github.com/rawtreedb/datathon-fme-2026-simulator
- Repositorios de los mejores equipos: pendientes (Marc los pasará más adelante).
## Resultado final
 
- **Posición 31 de 42 equipos.** Los equipos que quedaron por debajo estaban en rojo (sin modelo válido) menos uno, así que fuimos penúltimos entre los equipos con modelo válido.
- **Durante la práctica estuvimos casi todo el rato entre los puestos 22 y 24.**
- **Empezamos la ronda final en rojo.** Lanzamos solo la versión de Jorge y a los 5 minutos cayó en rojo por respuestas inválidas: fallaba en torno al 80 % porque recomendaba productos sin stock (out of stock).
- **Media hora sin modelo.** Tardamos unos 30 minutos en poner en marcha el modelo con el arreglo que dio Claude, y para entonces ya era tarde: esa media hora quedó registrada como si no tuviéramos ningún modelo.
- **El error evitable:** no pensamos en dejar corriendo el otro modelo mientras arreglábamos el nuevo. Con cualquier versión válida respondiendo, esa media hora habría puntuado.
- **No presentamos ante el jurado.** Nos frustró haber quedado tan abajo y que el modelo fallara nada más empezar. Eso supuso renunciar a los 30 puntos del jurado.
- **Podio:** Vamos Breakfast (58,3), Skore Money (57,7) y toievskithon (56,2). Cuatro equipos superaron al recomendador de referencia de los organizadores.
## El equipo
 
- Cuatro personas: Marc, Arnau (amigo de Marc), y Jorge y Pablo, dos matemáticos que conocimos allí.
- Nadie tenía una cuenta de pago con agente. Íbamos a tientas preguntando a chats de IA: Marc a Claude (y al principio algo a Gemini), Arnau sobre todo a DeepSeek, Pablo a ChatGPT y Jorge a Gemini.
- Marc pagó Claude Code el domingo por la mañana; hasta entonces no hubo ningún agente en el equipo.
- Pablo era el que mejor sacaba los modelos.
## Cronología
 
### Sábado por la mañana: elección del reto
 
- Presentación de los tres retos.
- El tercero no me apetecía: era más fácil y no me llamaba la atención.
- El primero iba sobre un chip de Arduino para Qualcomm (los de las CPU Snapdragon): me parecía guay pero más complejo.
- El segundo, este, me pareció interesante y no tan complejo. Como equipo elegimos el de TinyBird (RawTree).
### Sábado, hasta la 13:00 aprox.: entorno y baseline
 
- Empezamos a mirar cómo se ejecutaba.
- Al principio decidimos no compartir un repositorio (o tener uno que no tuviera el resto) e íbamos cambiando el código a mano, cada uno por su cuenta, para evitar pull requests: todos tocábamos el mismo fichero, `queries.sql`. No me quedó claro para qué lo usábamos ni cuándo se llamaba.
- Yo era el que tenía la terminal buena, porque estuve mucho rato configurándola para que el entorno funcionara. Los demás accedían a mi Visual Studio Code con Live Share.
- Tras un buen rato conseguimos ejecutar el SQL y el baseline. Sobre la 13:00, después de comer, el entorno ya estaba montado.
### Sábado, de 13:00 a 16:00 aprox.: primeros modelos
 
- Probamos un modelo nuevo que era un pelín mejor que el baseline.
- Decidimos que cada uno hiciera su modelo por su cuenta, porque íbamos muy atrás.
- Sobre las 16:00 habíamos lanzado uno o dos modelos.
### Sábado, desde las 16:00: repositorio compartido
 
- Compartí el repositorio y cada uno se hizo una rama para intentar mejorar los modelos.
- Íbamos muy perdidos.
### Sábado por la tarde y noche: variantes del LightGBM
 
- Primero trabajamos con LGBM (LightGBM, Light Gradient Boosting Machine).
- Hacia la noche hicimos más modelos, casi todos basados en el anterior que mejor iba.
- Arnau y yo intentábamos mejorar, cada uno por su cuenta, el LGBM de Jorge.
- Jorge y Pablo intentaron, también por su cuenta, leer la descripción de los productos nuevos para hacer un embedding y saber qué productos se parecen. Creo que no lo consiguieron.
- Jorge hizo una versión que llamó "Optimus Prime Plus", que funcionó mejor.
### Domingo por la mañana: cold start y ronda final
 
- Cuando volví, intenté mejorar mi versión para que tratara el cold start (con Claude Code).
- Al acercarse la hora, mi versión no se veía mucho mejor que la nueva de Jorge, así que decidimos no ejecutar las dos, sino solo la de Jorge.
- A los 5 minutos de empezar cayó en rojo por las respuestas inválidas (ver "Resultado final").
- Mi hipótesis: ninguno de los modelos había tenido en cuenta que un producto se podía quedar sin stock, quizá porque en las simulaciones no había ningún ejemplo de eso.
## Feedback de Marc (16 nov 2026)
 
### 1. No entendí el problema ni la herramienta
 
- No entendí qué era exactamente RawTree ni qué funcionalidades nos daba para mejorar el baseline.
- Tampoco entendí el framework del starter kit. Entendí más o menos el flujo, pero, aun sabiendo bastante Python, no llegué a entender de verdad qué estábamos haciendo.
- No tenía claro qué era lo que había que optimizar.
- No entendí por qué nos conectábamos a RawTree con la API key y, a la vez, la web explicaba comandos de RawTree (crear una database, etc.). Uno de los organizadores me habló de "conectar vuestro MCP", pero nosotros no teníamos agentes.
- Todo el rato pensaba: ¿no tendríamos que estar usando RawTree?, ¿no tendríamos que saber en qué nos está facilitando la vida?
### 2. Cómo trabajamos como equipo
 
- Cada uno fue más o menos por su cuenta, con algo de trabajo conjunto.
- Yo me encargué del entorno porque era quien mejor sabía hacerlo: buscar los requirements y arreglar la terminal para que todos pudiéramos ejecutar.
- Cuando conseguimos ejecutar el baseline, cada uno se puso a buscar una solución con la IA por separado, así que cada uno acabó con modelos distintos.
### 3. Ideas que salieron
 
- Usar el historial anterior de ventas como feature.
- Hacer un embedding de la descripción de cada producto, para que un producto nuevo se acerque a los que se le parecen.
- Agrupar los productos en clusters y mandar a cada usuario a un cluster de unos 20 productos, para sacar los candidatos de un conjunto más pequeño y ser más eficientes.
- El problema: todo esto lo hacíamos con la IA sin entender qué era el modelo ni qué había que optimizar.
### 4. Cómo medíamos los modelos
 
- Con la leaderboard: al entrar te mostraba tus versiones y sus estadísticas por ventanas de 15 minutos, 30 minutos, 1 hora y 3 horas.
- Nos fijábamos sobre todo en qué versión tenía mejor conversión y mejor cold start, y después en los € por request.
### 5. No sabíamos qué buscar
 
- No sabíamos cómo lo hacían tan bien los demás equipos.
- No sabíamos si la clave era cambiar LGBM por un algoritmo mucho más complicado o dar con una feature muy buena.
- En el fondo, no sabíamos qué pedirle a la IA.
### 6. Lo sabíamos, pero dimos por hecho que la IA lo había hecho
 
- Sabíamos que habría productos y usuarios nuevos, sin historial.
- Lo del stock también lo habíamos leído.
- Yo suponía que la IA ya lo había entendido y lo había tenido en cuenta, porque le habíamos pasado el repositorio y el enunciado del reto.
- También estaba la regla de no recomendar productos que el usuario ya hubiera comprado. Estaba escrita y los organizadores insistieron en ella, pero no llegamos a comprobar que nuestro código la aplicara.
### 7. La ronda final
 
- El enunciado avisaba de que a las 10:00 habría productos que no conocíamos y una demanda masiva, y que eso podía hacer fallar el programa.
- El nuestro falló desde el principio (ver "Resultado final").
### 8. Velocidad de iteración
 
- Creo que los otros equipos usaron agentes y varios modelos para iterar rápido.
- Nosotros tardábamos muchísimo solo en enviar un modelo nuevo: cogíamos el que ya habíamos visto que funcionaba más o menos en la leaderboard y le pedíamos a la IA que nos explicara cómo mejorarlo.
- Para no esperar 20–30 minutos a que se actualizara la leaderboard, conseguimos ejecutar el notebook de evaluación offline, que compara un modelo con el baseline. Pero ni siquiera eso lo entendimos, porque no vimos dónde se explicaba en el GitHub.
### 9. Cómo me sentí
 
- Entendía tan poco que tuve que pagar Claude y pedirle a la IA "un modelo mejor" sin saber qué pedir ni cómo plantearlo.
- Mucha impotencia y frustración por estar tan abajo, hasta el punto de no presentar.
## Mensajes de los organizadores (resumen)
 
### Cambio de regla: 5 segundos de gracia tras un agotamiento de stock (sábado, desde las 16:45)
 
- Cada 5 minutos se agotaban unos cuantos productos.
- Un agotamiento tardaba hasta unos 2 s en aparecer en `shop.catalog_updates`, pero las respuestas se validaban contra el stock en el instante exacto de la petición.
- Desde las 16:45, un producto agotado menos de 5 s antes de la petición ya no invalidaba la respuesta: ese hueco no vendía nada, pero el resto de la lista contaba.
- El fallo lo detectó y lo comunicó el `team_02`.
### Aclaración: compras repetidas
 
- En los datos (`shop.history_events`, `shop.history_sessions`, `shop.events`) los compradores sí repiten producto, y parecía contradecir el enunciado.
- Las dos cosas eran ciertas: el histórico y los eventos orgánicos se generaron sin ese efecto, pero **en lo que puntúa**, un comprador que ya compró un producto (en la tienda o por nuestras recomendaciones) es mucho menos probable que lo vuelva a comprar.
- Por tanto no había que deducir el comportamiento de recompra del histórico. El `feedback` propio sí seguía la regla.
- Consejo de los organizadores: fíate del bucle de feedback más que de la documentación, y de la documentación más que de un fichero de datos antiguo.
### Aviso de la ronda final (antes de las 10:00)
 
- Lista de comprobación: recomendador corriendo y respondiendo, portátil sin suspenderse, `.env` con la clave correcta. Bastantes equipos no estaban respondiendo.
- La final continuaba el mismo mundo: nada se reiniciaba a las 10:00.
- Las compras seguían contando: lo comprado en el histórico, en la tienda el sábado y por nuestras recomendaciones.
- Los carritos no: `add_to_cart` era solo actividad, sin carritos pendientes que se convirtieran en compra.
- El catálogo seguía: los productos añadidos el sábado continuaban, y los cambios de stock y precio seguían llegando.
### Cierre
 
- 42 equipos, 26 horas, 281.000 compradores atendidos, 1.423 versiones de modelo desplegadas, unos 33 millones de consultas en RawTree.
- Según los organizadores, los mejores lo consiguieron con **cold start, no repetir lo que el comprador ya había comprado y una fiabilidad a prueba de bombas**.
### El simulador es público
 
Repositorio: https://github.com/rawtreedb/datathon-fme-2026-simulator
 
- `world.py`: el modelo oculto. Cómo el gusto, el precio, la sesión, los pares "va con", las tendencias y el "ya comprado" se convertían en cada clic y compra.
- `simulator.py`: el flujo en vivo: peticiones, agotamientos, productos nuevos, bajadas de precio, tendencias, plazos y validación.
- `leaderboard/state.py`: cómo se calculaba la puntuación.
- El README: el modelo explicado término a término, las constantes, qué importaba más y las rarezas encontradas durante el evento.
- Incluye las semillas: se puede reconstruir el mismo catálogo, historial y ronda final, y reproducir nuestras propias respuestas para ver de dónde salieron los puntos.
## Lo que entendí después (explicación de Claude, 16 nov 2026)
 
### Qué se suponía que había que hacer
 
RawTree es una base de datos en la nube: metes JSON y lo consultas con SQL. En el reto hacía dos papeles a la vez.
 
1. **Era el buzón entre el equipo y el simulador.** Los organizadores escribían peticiones en `shop.requests`, el equipo escribía la respuesta en `team_31.submissions` y ellos devolvían qué pasó en `team_31.feedback`. El programa era un bucle: leer peticiones nuevas, elegir 10 productos, escribirlos, y repetir cada 0,3 s.
2. **Era donde estaban todos los datos para decidir qué recomendar**: catálogo, 60 días de historial, eventos en vivo y el feedback propio.
El trabajo real era el segundo punto: hacer preguntas a los datos con SQL hasta entender qué hace comprar a un cliente, y convertir esas respuestas en reglas o features. El ejemplo más claro: un producto ya visto se compraba 13 veces más. Eso no lo descubre un modelo mejor, lo descubre una consulta. Nosotros hicimos el camino al revés: primero el modelo (LightGBM, embeddings) y casi al final mirar los datos.
 
`queries.sql` no lo llamaba nadie: era una chuleta de consultas de ejemplo (catálogo actual, popularidad, co-vistas, leer el feedback) para copiar y adaptar.
 
### ¿Algoritmo más complicado o una feature muy buena?
 
Con lo que vimos en los datos, la respuesta es la segunda, y la intuición de Marc sobre RawTree era la correcta.
 
- **No era el algoritmo.** Todas las variantes de LightGBM del equipo quedaron entre el 0,8 % y el 1,35 %; cambiar el modelo movía décimas. La distancia con los primeros (2,2 %) era de otro orden.
- **Era saber qué buscaba el simulador.** Un comprador simulado sigue unas pocas reglas escondidas. El cierre de los organizadores lo confirma: `world.py` combina gusto, precio, sesión, pares "va con", tendencias y "ya comprado".
- **RawTree era la herramienta para encontrarlas.** La pregunta útil para la IA no era "dame un modelo mejor", sino "ayúdame a escribir consultas para ver en qué se diferencian las sesiones que acaban en compra de las que no".
- **La leaderboard dice cuál versión va mejor, no por qué.** Para el porqué estaba `team_31.feedback`, que casi no miramos hasta el final.
- **La idea de los clusters** apuntaba a la eficiencia, pero el tiempo no era el problema: respondíamos en 1 s de los 3 disponibles. Lo que faltaba era acertar con qué enseñar.
### Por qué había API key, comandos y MCP
 
Son tres puertas a la misma base de datos, y la API key es la llave de las tres:
 
- **La API desde Python** (`POST /v1/query`): la usa el recomendador mientras corre solo. Es la que usamos.
- **La CLI** (`rtree query "SELECT ..."`): para explorar a mano desde la terminal. Los comandos de la web (crear una database, etc.) eran de esto. Para el reto no hacía falta crear nada: ya existían `shop` y `team_31`.
- **El MCP**: para que un agente de IA consulte RawTree directamente.
Con un agente conectado por MCP, la IA lanza las consultas, lee el feedback y prueba ideas ella misma. Sin él, cada comprobación pasa por una persona que copia y pega: así trabajamos todo el sábado con los chats, y también el domingo con Claude Code, porque no llegamos a conectar el MCP. Por eso el análisis clave llegó a 20 minutos del final.
 
### Sobre el fallo del stock
 
- **El histórico no lo enseñaba**: los cambios de stock llegaban en vivo por `shop.catalog_updates` (`out_of_stock`, `back_in_stock`). Un modelo entrenado solo con el histórico no podía aprenderlo.
- **Pero sí había ejemplos en la práctica**: se agotaban productos cada 5 minutos desde el sábado, los organizadores cambiaron una regla por ello a las 16:45, y nuestro propio feedback registró unas 1.300 respuestas inválidas entre las 0:00 y las 2:00 del domingo.
- **No es un problema del modelo, sino de un filtro**: basta comprobar el stock justo antes de enviar. Las versiones v7 y v8 que hizo Claude sí lo comprobaban cada segundo; la de Jorge, por lo visto, no (deducción: Claude no vio su código).
- **Sin resolver:** por qué esa versión pasó de fallar poco en la práctica a fallar un 80 % en la final. Se puede comprobar reproduciendo la ronda con el simulador.
### Sobre dar por hecho que la IA lo había tenido en cuenta
 
- Pasarle a una IA el repositorio y el enunciado no garantiza que el código cumpla cada regla. Un chat genera lo que se le pide en ese mensaje; si la petición es "mejora el modelo", nadie está vigilando el stock.
- La única forma de saberlo es comprobarlo: una prueba que mire si alguna respuesta enviada contiene un producto agotado, inactivo o ya comprado, y una consulta periódica a los `fallback_reason` del feedback.
- También le pasó a Claude: la v8 quitaba lo ya comprado, pero solo miraba las compras de las últimas 4 horas en `shop.events`. Según la aclaración de los organizadores también contaban las del histórico y las hechas a través de nuestras recomendaciones, y Marc no llegó a pasarle ese mensaje durante el evento.
## Datos objetivos de la sesión con Claude (para contrastar)
 
- En la ronda de práctica, la mejor versión del equipo rondaba el 1,0–1,35 % de conversión; los primeros equipos estaban en torno al 2,2 %.
- Llegó a haber seis versiones del equipo respondiendo a la vez, así que la nota era una mezcla de todas.
- La mejora offline no se trasladó al directo: una versión con hit@10 de 0,50 frente a 0,39 del baseline sacó en vivo un 0,82 %, de las peores del equipo.
- La pista más útil apareció a 20 minutos de la ronda final, al mirar por primera vez el feedback real: un producto ya visto en la sesión se compraba unas 13 veces más en el histórico (6,5 % frente a 0,44 % en la posición 1) y unas 5 veces más en vivo, y el 66 % de las listas del equipo no incluía ninguno. Ese análisis eran dos consultas SQL que se podían haber hecho el sábado por la mañana.
- A las 9:59 del domingo, un minuto antes de la ronda final, se pararon todas las terminales, y a las 10:37 todavía se estaba preguntando cómo ejecutar el recomendador. Los cambios de última hora se hicieron sin margen para comprobarlos.
## Lecciones provisionales
 
- Antes de modelar, explorar los datos con SQL y leer el feedback propio.
- Pedirle a la IA ayuda para entender los datos, no "un modelo mejor".
- No dar por hecho que la IA ha aplicado una regla: escribir una comprobación para cada regla del enunciado.
- Leer los anuncios de los organizadores y pasárselos a quien (o a lo que) esté escribiendo el código.
- Repositorio compartido y ramas desde el minuto uno, no a las 16:00.
- Una sola línea de trabajo con tareas repartidas, en vez de cuatro personas mejorando el mismo modelo por separado.
- La fiabilidad va antes que el modelo: un modelo peor que responde siempre con productos válidos puntúa más que uno mejor que falla.
- Mirar los errores del feedback (respuestas inválidas o tardías) igual que se mira la conversión.
- Nunca parar la versión que funciona hasta que la nueva esté respondiendo bien.
- Congelar cambios un rato antes de la hora límite.
- Validar siempre la respuesta antes de enviarla: 10 productos distintos, activos y con stock.
- Si hay MCP y se dispone de un agente, conectarlo al principio.
- Presentar aunque el resultado sea malo: el jurado valoraba el método y lo aprendido, y eran 30 puntos.
## Objetivo
 
Cuando tenga tiempo, coger el repositorio del simulador (con las semillas se puede reproducir la ronda final) y los repositorios de los mejores equipos, para entender:
 
1. El framework: el flujo pull → predict → push, RawTree y el notebook de evaluación offline.
2. Qué se pedía realmente y cómo se puntuaba (`world.py` y `leaderboard/state.py`).
3. Cómo hacerlo mucho mejor en la siguiente hackathon.
## Pendiente de añadir
 
- Cómo se decidía qué versiones dejar corriendo por la noche y por qué acabaron seis a la vez.
- Lo visto en otros equipos y las diapositivas de cierre de los organizadores.
- Enlaces a los repositorios de los mejores equipos.