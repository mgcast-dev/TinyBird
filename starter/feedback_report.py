"""Resumen del feedback real del equipo. Solo lee; no cambia nada.

    python starter/feedback_report.py > feedback.txt
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
HOURS = 12

FB = f"""(SELECT request_id::String AS request_id, ts::DateTime64(3) AS ts,
       served_by::String AS served_by, fallback_reason::String AS fallback_reason,
       latency_ms::Float64 AS latency_ms, clicked::String AS clicked,
       purchased::String AS purchased, revenue_eur::Float64 AS revenue_eur,
       shown::Array(String) AS shown
FROM {TEAM}.feedback
WHERE request_id::String != '__init__' AND ts::DateTime64(3) > now() - INTERVAL {HOURS} HOUR)"""

AGG = """count() AS n, countIf(clicked != '') AS clics, countIf(purchased != '') AS compras,
       round(sum(revenue_eur), 2) AS ingresos"""

QUERIES = {
    "1. por hora y served_by": f"""
        SELECT toStartOfHour(ts) AS hora, served_by, fallback_reason, {AGG},
               round(avg(latency_ms)) AS lat_media, round(quantile(0.95)(latency_ms)) AS lat_p95
        FROM {FB} GROUP BY hora, served_by, fallback_reason ORDER BY hora, served_by""",
    "2. posicion del clic (solo served_by=team)": f"""
        SELECT indexOf(shown, clicked) AS posicion, count() AS clics, countIf(purchased != '') AS compras
        FROM {FB} WHERE served_by = 'team' AND clicked != ''
        GROUP BY posicion ORDER BY posicion""",
    "3. por version de modelo": f"""
        SELECT s.model_version AS version, {AGG}, min(f.ts) AS desde, max(f.ts) AS hasta
        FROM {FB} AS f
        INNER JOIN (SELECT request_id::String AS request_id, any(model_version::String) AS model_version
                    FROM {TEAM}.submissions GROUP BY request_id) AS s ON s.request_id = f.request_id
        WHERE f.served_by = 'team' GROUP BY version ORDER BY desde""",
    "4. por pagina y usuario nuevo": f"""
        SELECT r.page AS page, r.nuevo AS nuevo, {AGG}
        FROM {FB} AS f
        INNER JOIN (SELECT request_id::String AS request_id, context.page::String AS page,
                           is_new_user::Bool AS nuevo, length(recent_views::Array(String)) AS nv
                    FROM shop.requests WHERE ts::DateTime64(3) > now() - INTERVAL {HOURS + 1} HOUR) AS r
              ON r.request_id = f.request_id
        WHERE f.served_by = 'team' GROUP BY page, nuevo ORDER BY page, nuevo""",
    "5. por numero de productos vistos": f"""
        SELECT least(r.nv, 6) AS vistos, {AGG}
        FROM {FB} AS f
        INNER JOIN (SELECT request_id::String AS request_id, length(recent_views::Array(String)) AS nv
                    FROM shop.requests WHERE ts::DateTime64(3) > now() - INTERVAL {HOURS + 1} HOUR) AS r
              ON r.request_id = f.request_id
        WHERE f.served_by = 'team' GROUP BY vistos ORDER BY vistos""",
    "6. la compra estaba entre lo visto en la sesion?": f"""
        SELECT has(r.rv, f.purchased) AS comprado_ya_visto, indexOf(f.shown, f.purchased) AS posicion, count() AS compras
        FROM {FB} AS f
        INNER JOIN (SELECT request_id::String AS request_id, toString(recent_views) AS rv_txt,
                           extractAll(toString(recent_views), 'p_[0-9A-Za-z_]+') AS rv
                    FROM shop.requests WHERE ts::DateTime64(3) > now() - INTERVAL {HOURS + 1} HOUR) AS r
              ON r.request_id = f.request_id
        WHERE f.purchased != '' GROUP BY comprado_ya_visto, posicion ORDER BY comprado_ya_visto, posicion""",
    "7. ejemplo de 3 filas de feedback": f"SELECT * FROM {TEAM}.feedback ORDER BY ts DESC LIMIT 3",
    "8. ejemplo de 2 peticiones": "SELECT * FROM shop.requests ORDER BY ts DESC LIMIT 2",
}

for name, sql in QUERIES.items():
    print(f"\n=== {name} ===")
    try:
        r = requests.post(f"{API}/v1/query", json={"sql": sql}, headers=H, timeout=120)
        if not r.ok:
            print("ERROR", r.status_code, r.text[:400])
            continue
        for row in r.json()["data"]:
            print(json.dumps(row, ensure_ascii=False, default=str))
    except Exception as e:
        print("ERROR", type(e).__name__, e)
