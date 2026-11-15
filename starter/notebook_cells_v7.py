# =====================================================================================
# CELDAS DEL NOTEBOOK PARA v7  (offline_eval.ipynb, con el kernel arrancado en starter/)
# Antes: entrena el modelo de corte ->  python starter\training7.py --cut 2026-11-07
# =====================================================================================


# ---------- CELDA A: SUSTITUYE la celda que importaba recommender2 (la de MODEL_SUFFIX) ----------
import os, sys
os.environ["MODEL_SUFFIX"] = "_cut"
sys.path.insert(0, ".")
for m in ("recommender7", "features7"):
    sys.modules.pop(m, None)
import recommender7 as ours
from features7 import add_hourly, make_ctx, rank_request, top10, viewed_from_requests

# Los 200 más vendidos ANTES del corte (el baseline real, sin ver el futuro)
best = sql(f"""
    SELECT product_id::String AS product_id, count() AS n
    FROM shop.history_events
    WHERE event_type = 'purchase' AND ts::DateTime64(3) < toDateTime('{CUT}')
    GROUP BY product_id ORDER BY n DESC LIMIT 200""").product_id.tolist()

# Vistas por subcategoría y hora del periodo de test: la tendencia se calcula punto-en-el-tiempo
hv_rows = sql(f"""
    SELECT c.subcategory::String AS sub, toUnixTimestamp(toStartOfHour(e.ts::DateTime64(3))) AS h, count() AS v
    FROM shop.history_events AS e
    INNER JOIN shop.catalog AS c ON c.product_id::String = e.product_id::String
    WHERE e.event_type = 'view' AND e.ts::DateTime64(3) >= toDateTime('{CUT}') - INTERVAL 26 HOUR
    GROUP BY sub, h""")
add_hourly(ours.STATE, hv_rows.to_dict("records"))
print(len(hv_rows), "celdas horarias añadidas; modelo:", ours.MODELS["version"], "| exponente compra:", ours.MODELS["buy_exp"])


# ---------- CELDA B: SUSTITUYE la celda de políticas (baseline / coview / lgbm) ----------
test = test.sample(25000, random_state=0)      # 25k filas bastan y evitan esperar 15 min

def baseline(row):
    return best[:10]

def coview(row):
    score = Counter()
    for k, v in enumerate(reversed(row.recent_views or [])):
        for rank, q in enumerate(covisit.get(v["product_id"], [])):
            score[q] += 0.7 ** k / (rank + 1)
    items = [q for q, _ in score.most_common(10)]
    return (items + [p for p in best if p not in items])[:10]

def ours_policy(row):
    viewed = viewed_from_requests(row.recent_views)
    ctx = make_ctx(row.context, row.ts, state=ours.STATE)
    return top10(rank_request(ours.STATE, ours.MODELS, ctx, viewed, ours.ACTIVE), best)

_memo = {}                                      # cada política se calcula una sola vez por fila
def memo(name, f):
    def g(row):
        k = (name, row.Index)
        if k not in _memo:
            _memo[k] = f(row)
        return _memo[k]
    return g

POLICIES = {k: memo(k, f) for k, f in {"baseline": baseline, "coview": coview, "ours": ours_policy}.items()}

# (el resto de celdas queda igual: tabla c09, bootstrap con diff("ours", "baseline"))


# ---------- CELDA C: NUEVA, al final. Compras (lo que puntúa el reto) y usuarios fríos ----------
test["purchased"] = test.purchased.replace("", None)

def purchase_hit(policy, df):
    d = df[df.purchased.notna()]
    return float(np.mean([p in policy(r) for p, r in zip(d.purchased, d.itertuples())]))

cold = test[test.recent_views.apply(lambda v: len(viewed_from_requests(v)) == 0)]
print(f"usuarios fríos (sin vistas): {len(cold)} de {len(test)}")
out = {}
for name, p in POLICIES.items():
    out[name] = {"hit@10 compras (todos)": purchase_hit(p, test),
                 "hit@10 clics (todos)": evaluate(p, test)["hit_rate@10"],
                 "hit@10 clics (fríos)": evaluate(p, cold)["hit_rate@10"],
                 "hit@10 compras (fríos)": purchase_hit(p, cold)}
pd.DataFrame(out).T.round(4)
