"""¿Convierte más enseñar productos que el usuario ya ha visto en la sesión? Solo lee.

    python starter/feedback_report2.py > feedback2.txt
"""
import json
import os
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
for env in (HERE.parent / ".env", HERE / ".env", Path(".env")):
    for line in env.read_text().splitlines() if env.exists() else []:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())
API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ["TEAM_ID"]
H = {"Authorization": f"Bearer {os.environ['RAWTREE_API_KEY']}"}
HOURS = 14
RX = "'p_[0-9A-Za-z_]+'"

# feedback del equipo unido a las peticiones (rv = productos vistos, en orden)
J = f"""(SELECT f.shown AS shown, f.clicked AS clicked, f.purchased AS purchased, r.rv AS rv, r.nuevo AS nuevo
FROM (SELECT request_id::String AS request_id, clicked::String AS clicked, purchased::String AS purchased,
             shown::Array(String) AS shown
      FROM {TEAM}.feedback
      WHERE served_by::String = 'team' AND ts::DateTime64(3) > now() - INTERVAL {HOURS} HOUR) AS f
INNER JOIN (SELECT request_id::String AS request_id, is_new_user::Bool AS nuevo,
                   extractAll(toString(recent_views), {RX}) AS rv
            FROM shop.requests WHERE ts::DateTime64(3) > now() - INTERVAL {HOURS + 1} HOUR) AS r
      ON r.request_id = f.request_id
WHERE length(r.rv) > 0)"""

# lo mismo en el histórico (sistema antiguo)
HIST = f"""(SELECT shown::Array(String) AS shown, ifNull(clicked::String, '') AS clicked,
       ifNull(purchased::String, '') AS purchased, extractAll(toString(recent_views), {RX}) AS rv
FROM shop.history_sessions
WHERE ts::DateTime64(3) > now() - INTERVAL 25 DAY AND length(extractAll(toString(recent_views), {RX})) > 0)"""

PER_SLOT = """SELECT pos, has(rv, shown[pos]) AS visto, shown[pos] = rv[length(rv)] AS es_el_ultimo,
       count() AS veces, countIf(clicked = shown[pos]) AS clics, countIf(purchased = shown[pos]) AS compras
FROM {src} ARRAY JOIN arrayEnumerate(shown) AS pos
WHERE pos <= 5
GROUP BY pos, visto, es_el_ultimo ORDER BY pos, visto, es_el_ultimo"""

QUERIES = {
    "A. en vivo: peticiones segun si el slot 1 es un producto visto": f"""
        SELECT multiIf(shown[1] = rv[length(rv)], 'ultimo visto', has(rv, shown[1]), 'otro visto', 'no visto') AS slot1,
               nuevo, count() AS n, countIf(clicked != '') AS clics, countIf(purchased != '') AS compras,
               countIf(purchased != '' AND has(rv, purchased)) AS compras_de_vistos
        FROM {J} GROUP BY slot1, nuevo ORDER BY slot1, nuevo""",
    "B. en vivo: peticiones segun cuantos vistos hay en la lista": f"""
        SELECT least(length(arrayIntersect(shown, rv)), 4) AS vistos_en_lista, count() AS n,
               countIf(clicked != '') AS clics, countIf(purchased != '') AS compras
        FROM {J} GROUP BY vistos_en_lista ORDER BY vistos_en_lista""",
    "C. en vivo: por slot (1-5), producto visto o no": PER_SLOT.format(src=J),
    "D. historico: por slot (1-5), producto visto o no": PER_SLOT.format(src=HIST),
    "E. historico: compras de productos vistos frente al total": f"""
        SELECT count() AS sesiones, countIf(purchased != '') AS compras,
               countIf(purchased != '' AND has(rv, purchased)) AS compras_de_vistos,
               countIf(length(arrayIntersect(shown, rv)) > 0) AS sesiones_con_algun_visto_mostrado
        FROM {HIST}""",
}

for name, sql in QUERIES.items():
    print(f"\n=== {name} ===")
    try:
        r = requests.post(f"{API}/v1/query", json={"sql": sql}, headers=H, timeout=180)
        if not r.ok:
            print("ERROR", r.status_code, r.text[:400])
            continue
        for row in r.json()["data"]:
            print(json.dumps(row, ensure_ascii=False, default=str))
    except Exception as e:
        print("ERROR", type(e).__name__, e)
