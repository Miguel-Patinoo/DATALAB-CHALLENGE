"""Generador del tablero interactivo IPAP-BMC (HTML autocontenido, sin internet).
Uso: ver el final del archivo / celda de ejemplo para el notebook."""
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

def _corr(ccf_res, granger):
    """Convierte los resultados de correlación del notebook (ccf_res y granger) en tablas, gráficas y texto."""
    if isinstance(granger, pd.DataFrame): granger = granger.to_dict("index")
    ccf_res, granger = ccf_res or {}, granger or {}
    filas, graf, sigs = [], [], []
    for v, r in ccf_res.items():
        ks, pw = np.asarray(r["ks"]), np.asarray(r["pw"], float); bp = float(r["band_pw"]); m = (ks >= 0) & (np.abs(pw) > bp)
        arma = r.get("arma"); arma = f"({arma[0]},{arma[1]})" if arma else "—"
        filas.append(dict(v=v, d=r.get("d_x"), arma=arma, mx=round(float(np.abs(pw[ks >= 0]).max()), 3),
                          rez="; ".join(f"{int(k)} ({c:.3f})" for k, c in zip(ks[m], pw[m])) or "ninguno"))
        graf.append(dict(v=v, d=r.get("d_x"), arma=arma, ks=ks.tolist(), raw=_L(r["raw"]), pw=_L(pw), br=round(float(r["band_raw"]), 3), bp=round(bp, 3)))
        if m.any(): sigs.append((v, [int(k) for k in ks[m]]))
    gr = [dict(v=v, l=int(g["mejor rezago"]), p=float(g["p-valor"])) for v, g in granger.items()]
    gs = [g["v"] for g in gr if g["p"] < .05]
    if not ccf_res: return filas, gr, graf, "", ""
    t1 = ("Con la CCF blanqueada (banda ±95 %), muestran correlación significativa con el IPAP (variable adelantada k meses): "
          + "; ".join(f"{v} en los rezagos {', '.join(map(str, k))}" for v, k in sigs) + ". ") if sigs else "Con la CCF blanqueada ninguna variable supera la banda del 95 %. "
    t2 = (f"El test de Granger (α = 5 %) es significativo para: {', '.join(gs)}." if gs else "El test de Granger (α = 5 %) no es significativo para ninguna variable.") if gr else ""
    vars_ = ", ".join(ccf_res)
    b = (f"Se evaluaron variables externas ({vars_}). " + (f"Mostraron relación estadísticamente significativa con el IPAP: {', '.join(v for v, _ in sigs)}" if sigs else "Ninguna mostró relación significativa")
         + ", pero una relación significativa no garantiza un mejor pronóstico.")
    return filas, gr, graf, t1 + t2, b


# --- Aclaración sobre las redes neuronales (LSTM y GRU) -------------------------------------------------
# Se muestra automáticamente en la pestaña de cada red y en el Resumen. Para ponérsela a otro modelo, agrega
# la clave  aviso="texto"  en su entrada del diccionario `modelos`; con  aviso=""  se quita.
REDES_NEURONALES = {"LSTM", "GRU"}


def _es_red(nom):
    return str(nom).strip().upper() in REDES_NEURONALES


def _aviso(nombres):
    n = " y ".join(nombres)
    return (f"Las predicciones de {n} no son significativas ni confiables. Con la escasa cantidad de datos disponibles, "
            "las redes neuronales no logran capturar realmente la tendencia de la serie, por lo que sus resultados tienen "
            "mucha variabilidad al ejecutar el código, incluso fijando la misma semilla. Esto podría cambiar en el futuro, "
            "con mayor disponibilidad de datos.")


def _L(a, d=3):
    return [None if (v is None or pd.isna(v)) else round(float(v), d) for v in a]


def _acf(v, k=24):
    v = np.asarray(v, float); v = v - v.mean(); d = (v * v).sum()
    return [float((v[:-j] * v[j:]).sum() / d) for j in range(1, k + 1)]


def _pacf(v, k=24):  # Durbin-Levinson
    r = [1.0] + _acf(v, k); phi = np.zeros((k + 1, k + 1)); phi[1, 1] = r[1]; out = [r[1]]
    for m in range(2, k + 1):
        num = r[m] - sum(phi[m - 1, j] * r[m - j] for j in range(1, m))
        den = 1 - sum(phi[m - 1, j] * r[j] for j in range(1, m))
        phi[m, m] = num / den
        for j in range(1, m):
            phi[m, j] = phi[m - 1, j] - phi[m, m] * phi[m - 1, m - j]
        out.append(phi[m, m])
    return out


def _prueba_est(v, nombre):
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
        a = adfuller(v)[1]; k = kpss(v, regression="c", nlags="auto")[1]
    except Exception:
        return dict(n=nombre, adf=None, kpss=None, c="n/d")
    c = "Estacionaria" if (a < .05 and k >= .05) else "No estacionaria" if (a >= .05 and k < .05) else "Evidencia mixta"
    return dict(n=nombre, adf=round(a, 4), kpss=round(k, 4), c=c)


def _img(ruta):
    if not ruta: return None
    if str(ruta).startswith("data:"): return ruta
    import base64
    with open(ruta, "rb") as fh: return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def _metricas(y, p, lo=None, hi=None):
    y, p = np.asarray(y, float), np.asarray(p, float); e = y - p
    m = dict(rmse=float(np.sqrt((e ** 2).mean())), mae=float(np.abs(e).mean()), mape=float((np.abs(e) / np.abs(y)).mean() * 100), cob=None)
    if lo is not None:
        m["cob"] = float(((y >= np.asarray(lo)) & (y <= np.asarray(hi))).mean() * 100)
    return m


def construir_tablero(serie, modelos, ruta="tablero_ipap.html", ventana=36, ccf_res=None, granger=None, ccf_img=None,
                      titulo="IPAP-BMC · Tablero de pronóstico a 12 meses"):
    """serie: pd.Series mensual del IPAP (índice fecha).
    modelos: {nombre: dict(desc=str, params=str, banda='IC 90 %', prueba=DataFrame[ds,y,yhat(,yhat_lower,yhat_upper)],
              pronostico=DataFrame[Fecha,Pronóstico, <inf>, <sup>], color='#hex', aviso=str opcional,
              validacion=str opcional (cómo se eligió/validó), usa_prueba=bool opcional (la prueba intervino en la elección),
              salvedades=list[str] opcional (aclaraciones propias del modelo))}  (inf/sup: 2 últimas columnas)
    ccf_img: (opcional) ruta de un PNG con las gráficas CCF; solo se usa si no se entrega ccf_res (que dibuja las gráficas nativas).
    LSTM y GRU llevan automáticamente la aclaración de que sus predicciones no son confiables (ver _aviso)."""
    s = serie.dropna().astype(float).copy(); s.index = pd.to_datetime(s.index)
    fut = pd.to_datetime(next(iter(modelos.values()))["pronostico"]["Fecha"])
    x = [d.strftime("%Y-%m") for d in list(s.index) + list(fut)]; pos = {l: i for i, l in enumerate(x)}

    def al(f, v):
        a = [None] * len(x)
        for fi, vi in zip(pd.to_datetime(f), v):
            if pd.notna(vi): a[pos[fi.strftime("%Y-%m")]] = round(float(vi), 3)
        return a

    real = _L(s.values) + [None] * len(fut); n0 = len(x) - ventana - len(fut)
    # --- Descriptiva ---
    d = s.describe()
    stats = [("Observaciones", int(d["count"])), ("Periodo", f"{x[0]} → {x[len(s)-1]}"), ("Media", d["mean"]), ("Desv. estándar", d["std"]),
             ("Mínimo", d["min"]), ("Mediana", d["50%"]), ("Máximo", d["max"]), ("Asimetría", s.skew()), ("Curtosis", s.kurt())]
    dif = s.diff().dropna()
    ac = {k: dict(acf=_L(_acf(v)), pacf=_L(_pacf(v)), band=round(1.96 / np.sqrt(len(v)), 3)) for k, v in [("nivel", s.values), ("dif", dif.values)]}
    ccf_f, gr_f, ccf_g, txt_corr, txt_exog = _corr(ccf_res, granger)
    # --- Modelos ---
    t0 = pd.to_datetime(next(iter(modelos.values()))["prueba"]["ds"]).min()
    naive = s[s.index < t0].iloc[-1]; pr0 = next(iter(modelos.values()))["prueba"]
    met_naive = _metricas(pr0["y"], np.full(len(pr0), naive))
    paleta = ["#e8590c", "#6741d9", "#0ca678", "#1c7ed6", "#d6336c", "#f08c00"]; out = []
    for i, (nom, m) in enumerate(modelos.items()):
        pr, pf = m["prueba"], m["pronostico"]; pf = pf.copy(); ci = [c for c in pf.columns if "inf" in str(c) or "P5" in str(c)][0]; cs = [c for c in pf.columns if "sup" in str(c) or "P95" in str(c)][0]
        lo = pr["yhat_lower"] if "yhat_lower" in pr else None; hi = pr["yhat_upper"] if "yhat_upper" in pr else None
        k = _metricas(pr["y"], pr["yhat"], lo, hi)
        res = (f"En los 12 meses de prueba el modelo obtuvo RMSE {k['rmse']:.2f}, MAE {k['mae']:.2f} y MAPE {k['mape']:.2f} %, "
               f"frente a {met_naive['mape']:.2f} % del baseline Naïve. " + (f"Cobertura del IC 90 % en prueba: {k['cob']:.0f} %. " if k["cob"] is not None else "")
               + m.get("resultado", ""))
        out.append(dict(n=nom, c=m.get("color", paleta[i % 6]), av=m.get("aviso", _aviso([nom]) if _es_red(nom) else ""), desc=m["desc"], params=m.get("params", ""), banda=m.get("banda", "IC 90 %"), res=res, k=k,
                        val=m.get("validacion", ""), up=bool(m.get("usa_prueba", False)), sv=list(m.get("salvedades", [])),
                        be=str(m.get("banda", "")).startswith("Banda emp"),
                        prueba=[dict(f=pd.Timestamp(a).strftime("%Y-%m"), y=float(b), p=float(c), e=float(b - c)) for a, b, c in zip(pr["ds"], pr["y"], pr["yhat"])],
                        pron=[dict(f=pd.Timestamp(a).strftime("%Y-%m"), p=float(b), lo=float(c), hi=float(e)) for a, b, c, e in zip(pf["Fecha"], pf["Pronóstico"], pf[ci], pf[cs])],
                        ch=dict(test=al(pr["ds"], pr["yhat"]), tlo=al(pr["ds"], lo) if lo is not None else None, thi=al(pr["ds"], hi) if hi is not None else None,
                                fc=al(pf["Fecha"], pf["Pronóstico"]), flo=al(pf["Fecha"], pf[ci]), fhi=al(pf["Fecha"], pf[cs]))))
    ts = pd.to_datetime(pr0["ds"]); fm = lambda d: pd.Timestamp(d).strftime("%Y-%m")
    meta = dict(tr=fm(s.index[s.index < t0][-1]), pi=fm(ts.min()), pf=fm(ts.max()), fi=fm(fut.min()), ff=fm(fut.max()), gen=pd.Timestamp.today().strftime("%d/%m/%Y"))
    tabla = [dict(n=o["n"], **{q: o["k"][q] for q in ("rmse", "mae", "mape")}) for o in out] + [dict(n="Naïve (baseline)", **{q: met_naive[q] for q in ("rmse", "mae", "mape")})]
    mejor = min([o for o in out if not o["av"]] or out, key=lambda o: o["k"]["mape"])["n"]   # las redes con aviso no compiten por "mejor"
    redes = [o["n"] for o in out if o["av"]]
    # --- Salvedades globales (se recalculan con cada corrida) ---
    y_fin = float(pr0.sort_values("ds")["y"].iloc[-1]); var = (y_fin / naive - 1) * 100
    lst = lambda v: ", ".join(v[:-1]) + " y " + v[-1] if len(v) > 1 else v[0]
    up = [o["n"] for o in out if o["up"]]; emp = [o["n"] for o in out if o["be"]]
    salv = [f"La prueba es una sola ventana de {len(pr0)} meses ({meta['pi']} a {meta['pf']}), en la que el IPAP {'subió' if var >= 0 else 'bajó'} "
            f"{abs(var):.1f} % frente al último dato de entrenamiento. Un modelo que proyecta crecimiento sale favorecido si la ventana fue de alza, "
            "y uno que proyecta estabilidad si fue de calma: el ranking por MAPE en prueba es una referencia, no una garantía de qué modelo rendirá mejor adelante."]
    if any(o["val"] for o in out):
        salv.append("No todos los modelos se eligieron con el mismo procedimiento (ver la tabla): solo el error en la prueba es comparable entre todos.")
    if up:
        salv.append(f"En {lst(up)} la prueba intervino en la elección de parámetros o de especificación, así que su error en prueba puede ser optimista frente al error real futuro.")
    if emp:
        salv.append(f"En {lst(emp)} la banda del 90 % es empírica: se construye con errores pasados de validación, no es un intervalo de predicción del modelo. "
                    "Es una aproximación y su cobertura real puede ser menor al 90 %.")
    D = dict(titulo=titulo, meta=meta, x=x, n0=n0, real=real, dif=_L(list(dif.values) + [None] * len(fut)) if False else _L([None] + list(dif.values) + [None] * len(fut)),
             stats=[(a, (round(float(b), 3) if not isinstance(b, (int, str)) else b)) for a, b in stats], ac=ac, 
             ccf=ccf_f, gr=gr_f, cg=ccf_g, ccfimg=_img(ccf_img), txtcorr=txt_corr, exogtxt=txt_exog, modelos=out, tabla=tabla, mejor=mejor, aviso=_aviso(redes) if redes else "",
             salv=salv, up=up, vtab=[dict(n=o["n"], v=o["val"], up=o["up"]) for o in out])
    html = _HTML.replace("__TITLE__", titulo).replace("__DATA__", json.dumps(D, ensure_ascii=False))
    with open(ruta, "w", encoding="utf-8") as f: f.write(html)
    print("Tablero guardado en:", ruta)
    return ruta


_HTML = r"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
:root{--bg:#f4f7fb;--card:#fff;--ink:#1d2a3a;--mut:#6b7a90;--line:#e3e9f1;--pri:#1c7ed6;--ok:#0ca678;--okbg:#e6fcf5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{background:linear-gradient(120deg,#1c3d5a,#1c7ed6);color:#fff;padding:26px 32px}header h1{margin:0;font-size:24px}header p{margin:4px 0 0;opacity:.85}
nav{display:flex;gap:8px;flex-wrap:wrap;padding:14px 32px;background:#fff;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}
nav button{border:1px solid var(--line);background:#fff;padding:8px 16px;border-radius:999px;cursor:pointer;font:inherit;color:var(--mut);transition:.15s}
nav button:hover{border-color:var(--pri);color:var(--pri)}nav button.on{background:var(--pri);border-color:var(--pri);color:#fff;font-weight:600}
main{max-width:1180px;margin:22px auto;padding:0 20px}section{display:none}section.on{display:block}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin-bottom:18px;box-shadow:0 2px 8px rgba(30,60,100,.05)}
.card h3{margin:0 0 10px;font-size:16px}.grid{display:grid;gap:18px}.g2{grid-template-columns:repeat(auto-fit,minmax(420px,1fr))}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:18px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;border-top:3px solid var(--c,var(--pri))}
.kpi small{color:var(--mut);display:block}.kpi b{font-size:21px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-weight:600;background:#f8fafd}tr.best td{background:var(--okbg);font-weight:700;color:#087f5b}.sc{max-height:340px;overflow:auto}
p.m{color:var(--mut);margin:6px 0 0}.tag{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;background:#edf2f7;color:var(--mut)}.tag.ok{background:var(--okbg);color:#087f5b}.tag.no{background:#fff0f0;color:#c92a2a}
.cw{position:relative}svg{width:100%;height:auto;display:block}.g{stroke:#e8edf4}.t{fill:#8795a8;font-size:11px}.tip{position:absolute;pointer-events:none;background:#1d2a3a;color:#fff;padding:6px 10px;border-radius:8px;font-size:12px;display:none;white-space:nowrap;z-index:3}
.lg{display:flex;gap:14px;flex-wrap:wrap;font-size:12.5px;color:var(--mut);margin-top:6px}.lg i{display:inline-block;width:14px;height:3px;margin-right:5px;vertical-align:middle;border-radius:2px}
.hero{background:linear-gradient(120deg,#e6fcf5,#e7f5ff);border:1px solid #c3fae8}
.card.fc{display:flex;flex-direction:column}.fh{flex:1 1 0;min-height:340px;position:relative;overflow:hidden}
.warn{background:#fff9db;border:1px solid #ffd43b}.warn h3{color:#b35c00}.warn p{margin:0}
.ccf{width:100%;border:1px solid var(--line);border-radius:10px;margin-top:8px}ul.u{margin:6px 0 0;padding-left:20px}ul.u li{margin:5px 0}.sub{font-weight:700;margin:14px 0 2px;color:#1c3d5a}
</style></head><body><header><h1>__TITLE__</h1><p>Validación hold-out: entrenamiento hasta jun-2025 · prueba jul-2025 a jun-2026 · pronóstico final con toda la historia</p></header>
<nav id="nav"></nav><main id="main"></main><script>
const D=__DATA__;const f=(v,d=2)=>v==null||isNaN(v)?'—':Number(v).toLocaleString('es-CO',{minimumFractionDigits:d,maximumFractionDigits:d});
const el=(h)=>{const t=document.createElement('div');t.innerHTML=h.trim();return t.firstChild};
function line0(host,o){const W=o.W||900,H=o.h||310,m={l:50,r:12,t:12,b:30},a=o.a||0,x=o.x.slice(a),n=x.length,sl=v=>v.slice(a);let ys=[];
 o.s.forEach(s=>sl(s.p).forEach(v=>v!=null&&ys.push(v)));(o.b||[]).forEach(b=>sl(b.lo).concat(sl(b.hi)).forEach(v=>v!=null&&ys.push(v)));
 let lo=Math.min(...ys),hi=Math.max(...ys),pd=(hi-lo)*.08||1;lo-=pd;hi+=pd;const X=i=>m.l+i*(W-m.l-m.r)/(n-1),Y=v=>H-m.b-(v-lo)*(H-m.t-m.b)/(hi-lo);
 let g=`<svg viewBox="0 0 ${W} ${H}">`;for(let k=0;k<=4;k++){const v=lo+(hi-lo)*k/4;g+=`<line class="g" x1="${m.l}" x2="${W-m.r}" y1="${Y(v)}" y2="${Y(v)}"/><text class="t" style="font-size:${o.fs||15}px" x="${m.l-6}" y="${Y(v)+4}" text-anchor="end">${f(v,0)}</text>`}
 const st=Math.ceil(n/9);for(let i=0;i<n;i+=st)g+=`<text class="t" style="font-size:${o.fs||15}px" x="${X(i)}" y="${H-9}" text-anchor="middle">${x[i]}</text>`;
 (o.b||[]).forEach(b=>{const L=sl(b.lo),U=sl(b.hi),id=[];L.forEach((v,i)=>v!=null&&id.push(i));if(!id.length)return;
  g+=`<path d="M${id.map(i=>X(i)+','+Y(U[i])).join('L')}L${id.slice().reverse().map(i=>X(i)+','+Y(L[i])).join('L')}Z" fill="${b.c}" opacity=".16"/>`});
 if(o.v!=null)g+=`<line x1="${X(o.v-a)}" x2="${X(o.v-a)}" y1="${m.t}" y2="${H-m.b}" stroke="#8795a8" stroke-dasharray="5 4"/>`;
 o.s.forEach(s=>{let d='',up=false;sl(s.p).forEach((v,i)=>{if(v==null){up=false;return}d+=(up?'L':'M')+X(i)+','+Y(v);up=true});
  g+=`<path d="${d}" fill="none" stroke="${s.c}" stroke-width="${s.w||2.2}" ${s.d?'stroke-dasharray="'+s.d+'"':''} stroke-linejoin="round"/>`;
  if(s.dots)sl(s.p).forEach((v,i)=>v!=null&&(g+=`<circle cx="${X(i)}" cy="${Y(v)}" r="3.2" fill="${s.c}"/>`))});
 g+=`<line id="hv" y1="${m.t}" y2="${H-m.b}" stroke="#adb5bd" style="display:none"/><rect x="${m.l}" y="0" width="${W-m.l-m.r}" height="${H}" fill="transparent" id="ov"/></svg>`;
 host.innerHTML=`<div class="cw">${g}<div class="tip"></div></div><div class="lg">${o.s.map(s=>`<span><i style="background:${s.c}"></i>${s.n}</span>`).join('')}${(o.b||[]).map(b=>`<span><i style="background:${b.c};opacity:.35;height:9px"></i>${b.n}</span>`).join('')}</div>`;
 const svg=host.querySelector('svg'),tip=host.querySelector('.tip'),hv=host.querySelector('#hv');hv.removeAttribute('id');
 host.querySelector('#ov').onmousemove=e=>{const r=svg.getBoundingClientRect(),px=(e.clientX-r.left)*W/r.width,i=Math.max(0,Math.min(n-1,Math.round((px-m.l)*(n-1)/(W-m.l-m.r))));
  hv.setAttribute('x1',X(i));hv.setAttribute('x2',X(i));hv.style.display='';
  tip.innerHTML=`<b>${x[i]}</b><br>`+o.s.map(s=>{const v=sl(s.p)[i];return v==null?'':`<span style="color:${s.c}">●</span> ${s.n}: ${f(v,o.dd??1)}<br>`}).join('')+(o.b||[]).map(b=>{const l=sl(b.lo)[i];return l==null?'':`${b.n}: ${f(l,1)} – ${f(sl(b.hi)[i],1)}`}).join('<br>');
  tip.style.display='block';tip.style.left=Math.min(e.clientX-r.left+14,r.width-170)+'px';tip.style.top=(e.clientY-r.top-10)+'px'};
 host.querySelector('#ov').onmouseleave=()=>{tip.style.display='none';hv.style.display='none'}}
function line(host,o){if(!o.fill)return line0(host,o);
 host.classList.add('fh');let lw=0,lh=0;
 const draw=()=>{const w=host.clientWidth,h=host.clientHeight;if(!w||!h||(w==lw&&h==lh))return;lw=w;lh=h;
  line0(host,Object.assign({},o,{W:w,h:Math.max(220,h-56)}));const lg=host.querySelector('.lg');
  if(lg){const hh=Math.max(220,h-lg.offsetHeight-6);if(Math.abs(hh-(h-56))>4)line0(host,Object.assign({},o,{W:w,h:hh}))}};
 new ResizeObserver(draw).observe(host);draw()}
function violin(host,v){const W=420,H=340,m={l:52,r:16,t:14,b:26},n=v.length,s=v.slice().sort((a,b)=>a-b),q=p=>{const i=(n-1)*p,l=Math.floor(i);return s[l]+(s[Math.min(l+1,n-1)]-s[l])*(i-l)},
 mean=s.reduce((a,b)=>a+b,0)/n,sd=Math.sqrt(s.reduce((a,b)=>a+(b-mean)**2,0)/(n-1)),bw=sd*Math.pow(n,-.2),lo=s[0],hi=s[n-1],pad=(hi-lo)*.06,
 Y=u=>H-m.b-(u-(lo-pad))*(H-m.t-m.b)/(hi-lo+2*pad),G=140,pts=[];let mx=0;
 for(let i=0;i<=G;i++){const u=lo+(hi-lo)*i/G;let d=0;for(const x of s)d+=Math.exp(-.5*((u-x)/bw)**2);d/=n*bw*Math.sqrt(2*Math.PI);pts.push([u,d]);mx=Math.max(mx,d)}
 const cx=(W+m.l-m.r)/2,hw=(W-m.l-m.r)*.4,R=d=>d/mx*hw;let g=`<svg viewBox="0 0 ${W} ${H}">`;
 for(let k=0;k<=4;k++){const u=lo-pad+(hi-lo+2*pad)*k/4;g+=`<line class="g" x1="${m.l}" x2="${W-m.r}" y1="${Y(u)}" y2="${Y(u)}"/><text class="t" style="font-size:12px" x="${m.l-6}" y="${Y(u)+4}" text-anchor="end">${f(u,0)}</text>`}
 g+=`<path d="M${pts.map(([u,d])=>(cx+R(d))+','+Y(u)).join('L')}L${pts.slice().reverse().map(([u,d])=>(cx-R(d))+','+Y(u)).join('L')}Z" fill="#1c7ed6" fill-opacity=".35" stroke="#1c7ed6" stroke-width="1.6"><title>Densidad estimada (kernel gaussiano)</title></path>`;
 g+=`<line x1="${cx}" x2="${cx}" y1="${Y(hi)}" y2="${Y(lo)}" stroke="#1d2a3a" stroke-width="1.5"/><rect x="${cx-8}" y="${Y(q(.75))}" width="16" height="${Y(q(.25))-Y(q(.75))}" fill="#1d2a3a"><title>Q1 ${f(q(.25),1)} · Q3 ${f(q(.75),1)}</title></rect><circle cx="${cx}" cy="${Y(q(.5))}" r="4" fill="#fff"><title>Mediana ${f(q(.5),1)}</title></circle>`;
 g+=`<text class="t" style="font-size:12px" x="${cx}" y="${H-8}" text-anchor="middle">IPAP-BMC (nivel)</text></svg>`;
 host.innerHTML=g;let modes=0;pts.forEach(([u,d],i)=>{if(i>0&&i<G&&d>pts[i-1][1]&&d>=pts[i+1][1]&&d>.1*mx)modes++});return{modes,q1:q(.25),q3:q(.75)}}
function cbars(host,ks,v,band,c,ttl){const W=440,H=170,m={l:40,r:8,t:18,b:22},n=ks.length,mx=Math.max(band*1.25,...v.map(u=>Math.abs(u||0))),X=i=>m.l+(i+.5)*(W-m.l-m.r)/n,Y=u=>m.t+(mx-u)/(2*mx)*(H-m.t-m.b);
 let g=`<svg viewBox="0 0 ${W} ${H}"><text class="t" x="${m.l}" y="11" style="font-weight:600">${ttl}</text>`;[-mx,0,mx].forEach(t=>g+=`<line class="g" x1="${m.l}" x2="${W-m.r}" y1="${Y(t)}" y2="${Y(t)}"/><text class="t" x="${m.l-5}" y="${Y(t)+4}" text-anchor="end">${f(t,2)}</text>`);
 const z=ks.indexOf(0);g+=`<line x1="${X(z)}" x2="${X(z)}" y1="${m.t}" y2="${H-m.b}" stroke="#adb5bd" stroke-dasharray="2 3"/>`;
 v.forEach((u,i)=>{if(u==null)return;g+=`<line x1="${X(i)}" x2="${X(i)}" y1="${Y(0)}" y2="${Y(u)}" stroke="${Math.abs(u)>band?'#e03131':c}" stroke-width="4"><title>k=${ks[i]}: ${f(u,3)}</title></line>`;if(ks[i]%12==0)g+=`<text class="t" x="${X(i)}" y="${H-6}" text-anchor="middle">${ks[i]}</text>`});
 [band,-band].forEach(b=>g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(b)}" y2="${Y(b)}" stroke="#e03131" stroke-dasharray="5 4"/>`);host.innerHTML=g+'</svg>'}
function bars(host,v,band,c){const W=440,H=190,m={l:36,r:8,t:10,b:22},n=v.length,X=i=>m.l+(i+.5)*(W-m.l-m.r)/n,Y=u=>m.t+(1-u)*(H-m.t-m.b)/2;
 let g=`<svg viewBox="0 0 ${W} ${H}">`;[-1,-.5,0,.5,1].forEach(t=>g+=`<line class="g" x1="${m.l}" x2="${W-m.r}" y1="${Y(t)}" y2="${Y(t)}"/><text class="t" x="${m.l-5}" y="${Y(t)+4}" text-anchor="end">${t}</text>`);
 v.forEach((u,i)=>{g+=`<line x1="${X(i)}" x2="${X(i)}" y1="${Y(0)}" y2="${Y(u)}" stroke="${Math.abs(u)>band?'#e03131':c}" stroke-width="5"><title>Rezago ${i+1}: ${f(u,3)}</title></line>`;if((i+1)%6==0)g+=`<text class="t" x="${X(i)}" y="${H-6}" text-anchor="middle">${i+1}</text>`});
 [band,-band].forEach(b=>g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(b)}" y2="${Y(b)}" stroke="#e03131" stroke-dasharray="5 4"/>`);host.innerHTML=g+'</svg>'}
const tb=(h,rows,best)=>`<table><tr>${h.map(x=>`<th>${x}</th>`).join('')}</tr>${rows.map((r,i)=>`<tr class="${i===best?'best':''}">${r.map(x=>`<td>${x}</td>`).join('')}</tr>`).join('')}</table>`;
const kpi=(l,v,c)=>`<div class="kpi" style="--c:${c||''}"><small>${l}</small><b>${v}</b></div>`;
const nav=document.getElementById('nav'),main=document.getElementById('main'),tabs=[];
function tab(name,html,after){const id='t'+tabs.length,s=el(`<section id="${id}">${html}</section>`);main.appendChild(s);const b=el(`<button>${name}</button>`);nav.appendChild(b);
 b.onclick=()=>{document.querySelectorAll('section,nav button').forEach(e=>e.classList.remove('on'));s.classList.add('on');b.classList.add('on')};tabs.push(b);after&&after(s)}
document.querySelector("header p").textContent=`Validación hold-out: entrenamiento hasta ${D.meta.tr} · prueba ${D.meta.pi} a ${D.meta.pf} · pronóstico final ${D.meta.fi} a ${D.meta.ff} con toda la historia · actualizado el ${D.meta.gen}`;
const A=D.x.length,nf=D.modelos[0].pron.length,last=A-nf-1,a0=D.n0,RN=new Set(D.modelos.filter(o=>o.av).map(o=>o.n)),mk=n=>n+(RN.has(n)?' ⚠':'');
// ---- Serie ----
tab('📊 Serie',`<div class="kpis">${D.stats.map(([k,v])=>kpi(k,typeof v==='number'?f(v,k=='Observaciones'?0:2):v)).join('')}</div>
<div class="grid g2"><div class="card"><h3>IPAP-BMC (nivel)</h3><div id="sn"></div></div><div class="card"><h3>Primera diferencia (t vs t-1)</h3><div id="sd"></div></div></div>
<div class="grid g2"><div class="card"><h3>Distribución del IPAP · violin plot</h3><div id="vp"></div></div><div class="card"><h3>¿Qué muestra la distribución?</h3><div id="vt"></div></div></div>
<div class="card"><h3>Estacionalidad</h3><p>En la ACF de la serie diferenciada, los rezagos 12 y 24 valen ${f(D.ac.dif.acf[11],2)} y ${f(D.ac.dif.acf[23],2)} (banda ±${f(D.ac.dif.band,2)}): el rezago 12 queda ${Math.abs(D.ac.dif.acf[11])>D.ac.dif.band?'fuera':'dentro'} de la banda y el 24 ${Math.abs(D.ac.dif.acf[23])>D.ac.dif.band?'fuera':'dentro'}. Según el análisis del notebook no se modela estacionalidad anual (Prophet usa <code>yearly_seasonality=False</code>).</p></div>
<div class="grid g2"><div class="card"><h3>ACF · serie diferenciada</h3><div id="a1"></div></div><div class="card"><h3>PACF · serie diferenciada</h3><div id="a2"></div></div>
<div class="card"><h3>ACF · nivel</h3><div id="a3"></div></div><div class="card"><h3>PACF · nivel</h3><div id="a4"></div></div></div>
<p class="m" style="margin:-6px 0 14px">Barras rojas: fuera de la banda ±95 %.</p>
<div class="card"><h3>Variables exógenas: correlación cruzada (CCF con prewhitening) y Granger</h3><div class="grid g2"><div>${tb(['Variable','d','ARMA','Rezagos significativos X→IPAP','Máx |corr|'],D.ccf.map(r=>[r.v,r.d,r.arma,r.rez,f(r.mx,3)]))}</div>
<div>${tb(['Variable','Mejor rezago','p-valor Granger'],D.gr.map(r=>[r.v,r.l,`<span class="tag ${r.p<.05?'ok':''}">${f(r.p,4)}</span>`]))}</div></div><p class="m">${D.txtcorr}</p>${D.cg.length||D.ccfimg?`<h3 style="margin-top:18px">Gráficas de correlación cruzada (CCF) entre cada variable y el IPAP</h3>${D.cg.length?'<div id="ccfg"></div>':`<img class="ccf" src="${D.ccfimg}" alt="CCF entre el IPAP y las variables exógenas">`}<p class="m"><b>Cómo leerlas:</b> izquierda, CCF de las series diferenciadas; derecha, CCF con prewhitening (quita la autocorrelación propia, por lo que es la prueba más exigente). Las líneas rojas son la banda del 95 %: una barra que la sobrepase es significativa. A la derecha de 0 (k &gt; 0) la variable externa antecede al IPAP, que es lo útil para pronosticar.</p>`:''}</div>`,
 s=>{const r=D.real,d=D.dif;{const vals=r.filter(u=>u!=null),vr=violin(s.querySelector('#vp'),vals),g=k=>(D.stats.find(z=>z[0]==k)||[])[1],md=g('Mediana'),mn=g('Media'),sk=g('Asimetría'),ku=g('Curtosis');
 s.querySelector('#vt').innerHTML=`<p>El violin muestra cómo se reparten los ${vals.length} valores observados del IPAP: donde es más ancho, el índice pasó más tiempo. Dentro se ve la caja (Q1–Q3: ${f(vr.q1,1)} a ${f(vr.q3,1)}) y la mediana (${f(md,1)}).</p>
 <ul class="u"><li>Media ${f(mn,1)} frente a mediana ${f(md,1)}: la media queda ${mn>md?'por encima':'por debajo'}, lo que va con una asimetría ${sk>0?'positiva (cola hacia valores altos)':'negativa (cola hacia valores bajos)'} de ${f(sk,2)}.</li>
 <li>Curtosis ${f(ku,2)}: distribución ${ku<0?'platicúrtica, más aplanada que una normal':'leptocúrtica, con colas más pesadas que una normal'}.</li>
 <li>${vr.modes>1?`Hay ${vr.modes} concentraciones de valores (varios picos), así que la forma se aleja de la campana normal.`:'Hay una sola concentración principal de valores.'}</li></ul>`}
 D.cg.forEach(g=>{const w=document.createElement('div');w.innerHTML=`<div class="sub">IPAP vs ${g.v} <span class="tag">d=${g.d} · ARMA${g.arma}</span></div><div class="grid g2"><div></div><div></div></div>`;s.querySelector('#ccfg').appendChild(w);const h=w.querySelectorAll('.g2>div');cbars(h[0],g.ks,g.raw,g.br,'#1c7ed6','Diferenciada');cbars(h[1],g.ks,g.pw,g.bp,'#1c7ed6','Con prewhitening')});line(s.querySelector('#sn'),{x:D.x,s:[{n:'IPAP',c:'#1c7ed6',p:r}],h:260});line(s.querySelector('#sd'),{x:D.x,s:[{n:'Δ IPAP',c:'#6741d9',p:d}],h:260,dd:2});
 [['#a1','dif','acf'],['#a2','dif','pacf'],['#a3','nivel','acf'],['#a4','nivel','pacf']].forEach(([q,k,t])=>bars(s.querySelector(q),D.ac[k][t],D.ac[k].band,'#1c7ed6'))});
// ---- Modelos ----
const chartModel=(host,o,h)=>line(host,{x:D.x,a:a0,v:last,h:h||330,s:[{n:'Real',c:'#1d2a3a',p:D.real},{n:o.n+' · prueba',c:o.c,p:o.ch.test,d:'2 4',dots:1},{n:o.n+' · pronóstico final',c:o.c,p:o.ch.fc,dots:1,w:2.8}],
 b:[{n:o.banda+' (pronóstico)',c:o.c,lo:o.ch.flo,hi:o.ch.fhi}].concat(o.ch.tlo?[{n:'IC 90 % (prueba)',c:o.c,lo:o.ch.tlo,hi:o.ch.thi}]:[])});
D.modelos.forEach(o=>tab(mk(o.n),`${o.av?`<div class="card warn"><h3>⚠️ Aclaración: resultados poco confiables</h3><p>${o.av}</p></div>`:''}<div class="kpis">${kpi('RMSE (prueba)',f(o.k.rmse),o.c)}${kpi('MAE (prueba)',f(o.k.mae),o.c)}${kpi('MAPE (prueba)',f(o.k.mape)+' %',o.c)}${o.k.cob!=null?kpi('Cobertura IC 90 %',f(o.k.cob,0)+' %',o.c):''}${o.n==D.mejor?kpi('Ranking','🏆 Mejor','#0ca678'):''}${o.av?kpi('Fiabilidad','⚠ Baja','#f08c00'):''}</div>
<div class="grid g2"><div class="card"><h3>Método</h3><p>${o.desc}</p>${o.params?`<p class="m"><b>Hiperparámetros:</b> ${o.params}</p>`:''}${o.val?`<p class="m"><b>Validación:</b> ${o.val}</p>`:''}</div><div class="card"><h3>Resultados</h3><p>${o.res}</p></div></div>
${o.sv.length?`<div class="card warn"><h3>⚠️ Salvedades de este modelo</h3><ul class="u">${o.sv.map(t=>`<li>${t}</li>`).join('')}</ul></div>`:''}
<div class="card"><h3>Predicción · ${o.n}</h3><div class="ch"></div><p class="m">Línea vertical: último dato observado (${D.x[last]}). Punteado: predicción sobre los 12 meses de prueba. ${o.be?'<b>La banda es empírica: sale de errores pasados de validación; no es un intervalo de predicción del modelo.</b>':o.banda.startsWith('IC')?'':'<b>La banda P5–P95 mide la variación entre semillas del entrenamiento; no es un intervalo de predicción.</b>'}</p></div>
<div class="grid g2"><div class="card"><h3>Prueba: real vs. predicho</h3><div class="sc">${tb(['Mes','Real','Predicho','Error'],o.prueba.map(r=>[r.f,f(r.y),f(r.p),f(r.e)]))}</div></div>
<div class="card"><h3>Pronóstico final (modelo con toda la historia)</h3><div class="sc">${tb(['Mes','Pronóstico',o.banda=='IC 90 %'?'IC 90 % inf':o.be?'Banda 90 % inf':'P5 inf',o.banda=='IC 90 %'?'IC 90 % sup':o.be?'Banda 90 % sup':'P95 sup'],o.pron.map(r=>[r.f,f(r.p),f(r.lo),f(r.hi)]))}</div></div></div>`,s=>chartModel(s.querySelector('.ch'),o)));
// ---- Resumen ----
function salvCard(){return `<div class="card"><h3>🔎 Cómo se validó cada modelo y qué tener en cuenta</h3>${D.vtab.some(r=>r.v)?tb(['Modelo','Cómo se eligió y validó','¿La prueba intervino en la elección?'],D.vtab.map(r=>[mk(r.n),r.v||'—',r.v?(r.up?'<span class="tag no">Sí · error de prueba posiblemente optimista</span>':'<span class="tag ok">No</span>'):'—'])):''}<ul class="u">${D.salv.map(t=>`<li>${t}</li>`).join('')}</ul></div>`}
function interp(){const b=D.modelos.find(o=>o.n==D.mejor),u=D.real[last],P=b.pron,p0=P[0].p,pn=P[P.length-1].p,mn=Math.min(...P.map(r=>r.lo)),mx=Math.max(...P.map(r=>r.hi)),
 hw=P.reduce((s,r)=>s+(r.hi-r.lo)/2,0)/P.length,pc=v=>(v/u-1)*100,sg=v=>(v>=0?'+':'')+f(v,1)+' %',err=b.k.mape/100*u,nv=D.tabla[D.tabla.length-1],
 okm=D.modelos.filter(o=>!o.av),fin=(okm.length?okm:D.modelos).map(o=>o.pron[o.pron.length-1].p),mean=P.reduce((s,r)=>s+r.p,0)/P.length,tend=Math.abs(pc(mean))<3?'relativamente estables':pc(mean)>0?'al alza':'a la baja';
 return `<div class="card"><h3>📘 ¿Cómo leer y usar esta información? Guía para el pequeño productor</h3>
 <div class="sub">1. Qué estamos mirando</div><p>El IPAP-BMC es un <b>índice general</b> de precios agropecuarios: si sube, los precios del campo en promedio están más altos; si baja, más bajos. No es el precio de un producto en particular, sino una señal del “clima” general del mercado.</p>
 <div class="sub">2. Qué dice el pronóstico</div><p>El último dato observado (${D.x[last]}) fue <b>${f(u,1)}</b>. El mejor modelo (${b.n}) proyecta que entre ${P[0].f} y ${P[P.length-1].f} el índice se mueva entre <b>${f(p0,1)}</b> y <b>${f(pn,1)}</b> (${sg(pc(p0))} y ${sg(pc(pn))} frente al último dato), es decir, precios <b>${tend}</b>${u>=0.97*Math.max(...D.real.filter(v=>v!=null))?' y cerca del máximo de toda la serie. Un nivel alto y estable no garantiza que seguirá así':''}.</p>
 <div class="sub">3. Qué tan confiable es</div><ul class="u"><li>En la prueba, el modelo se equivocó en promedio un <b>${f(b.k.mape,1)} %</b> (unos ${f(err,0)} puntos del índice). Repetir “igual que el último mes” se equivocó un ${f(nv.mape,1)} %.</li>
 <li>Use siempre el <b>rango del 90 %</b>, no la cifra puntual: ${b.be?'la banda es una aproximación construida con errores pasados, así que el valor real puede salirse de ella más de 1 de cada 10 veces.':'se espera que el valor real caiga dentro de la banda aproximadamente 9 de cada 10 veces.'} La banda es de unos ±${f(hw,0)} puntos y a lo largo del año va de ${f(mn,0)} a ${f(mx,0)}.</li>
 <li>Para ${P[P.length-1].f} los modelos${D.aviso?' confiables':''} van de ${f(Math.min(...fin),0)} a ${f(Math.max(...fin),0)}. Cuando los modelos no coinciden, hay más incertidumbre, sobre todo en los últimos meses del horizonte.${D.aviso?` ${[...RN].join(' y ')} no entran en este rango: sus predicciones no son confiables (ver la aclaración en el Resumen y en la pestaña de cada red).`:''}</li></ul>
 <div class="sub">4. Cómo puede usarlo</div><ul class="u"><li><b>Presupuestar con el rango, no con un punto:</b> use el límite inferior para cubrir costos en un escenario prudente y el superior como posible ganancia adicional.</li>
 <li><b>Ventas y siembras escalonadas:</b> si el escenario es de precios estables, evite concentrar toda la venta o toda la inversión en un solo momento.</li>
 <li><b>Compare cada mes</b> el dato real con la banda del 90 %. Si sale de la banda, el mercado cambió y el pronóstico debe revisarse antes de tomar decisiones grandes.</li>
 <li><b>Combínelo con lo que usted sabe:</b> sus costos, el precio local de su producto, sus contratos y el consejo de su cooperativa o asistente técnico. El índice orienta, pero no reemplaza su criterio.</li></ul>
 <div class="sub">5. Qué NO puede hacer este tablero</div><ul class="u"><li>No anticipa choques como sequías, heladas, plagas, paros de transporte o cambios de política; si ocurren, los precios pueden salirse de la banda.</li>
 <li>Entre más lejos el mes, más incierto: confíe más en los primeros meses que en los últimos.</li>
 ${D.exogtxt?`<li>${D.exogtxt}</li>`:''}
 <li>${D.up.length?`En ${D.up.join(', ')} los parámetros o la especificación se eligieron mirando los mismos 12 meses de prueba, así que el error real futuro podría ser algo mayor al reportado.`:'El error en prueba viene de una sola ventana de 12 meses; el error real futuro puede ser distinto.'}</li></ul>
 <p class="m">Esta es una guía de apoyo para entender los datos, no una garantía de precios ni asesoría financiera.</p></div>`}
const best=D.modelos.find(o=>o.n==D.mejor),TB=D.tabla.slice().sort((a,b)=>a.mape-b.mape),bi=TB.findIndex(r=>r.n==D.mejor);
tab('🏆 Resumen',`<div class="card hero"><h3>🏆 Mejor modelo: ${best.n}</h3><p>Menor MAPE en los 12 meses de prueba: <b>${f(best.k.mape)} %</b> (RMSE ${f(best.k.rmse)}), frente a ${f(D.tabla[D.tabla.length-1].mape)} % del baseline Naïve. Su pronóstico para ${best.pron[0].f} a ${best.pron[nf-1].f} va de ${f(best.pron[0].p,1)} a ${f(best.pron[nf-1].p,1)}.</p>
<p class="m">${D.up.includes(best.n)?'Nota: en este modelo la prueba intervino en la elección de parámetros o de especificación, por lo que su error de prueba puede ser algo optimista.':'Nota: la prueba es una sola ventana de 12 meses; ver las salvedades más abajo.'}</p></div>
${D.aviso?`<div class="card warn"><h3>⚠️ Aclaración sobre ${[...RN].join(' y ')}</h3><p>${D.aviso}</p></div>`:''}
<div class="card"><h3>Pronóstico del mejor modelo · ${best.n}</h3><div id="rb"></div></div>
<div class="grid g2"><div class="card"><h3>Métricas de error en prueba</h3>${tb(['Modelo','RMSE','MAE','MAPE (%)'],TB.map(r=>[(r.n==D.mejor?'🏆 ':'')+mk(r.n),f(r.rmse,3),f(r.mae,3),f(r.mape,3)]),bi)}<p class="m">Ordenados de menor a mayor MAPE. Menor es mejor. Verde: mejor modelo por MAPE${RN.size?' (entre los modelos confiables). ⚠ = resultado poco confiable, ver aclaración':''}.</p></div>
<div class="card fc"><h3>Pronóstico final: todos los modelos</h3><div id="rc"></div></div></div>
${salvCard()}
<div class="card"><h3>Pronóstico mes a mes</h3>${tb(['Mes'].concat(D.modelos.map(o=>mk(o.n))),D.modelos[0].pron.map((r,i)=>[r.f].concat(D.modelos.map(o=>f(o.pron[i].p)))))}</div>${interp()}`,
 s=>{chartModel(s.querySelector('#rb'),best,340);line(s.querySelector('#rc'),{x:D.x,a:a0,v:last,h:300,fill:1,fs:12,s:[{n:'Real',c:'#1d2a3a',p:D.real}].concat(D.modelos.map(o=>({n:mk(o.n),c:o.c,p:o.ch.fc,dots:1})))})});
tabs[0].click();
</script></body></html>"""


if __name__ == "__main__":
    print("Importa construir_tablero desde tu notebook (ver celda de ejemplo).")
