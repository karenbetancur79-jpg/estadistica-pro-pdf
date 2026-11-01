import io, itertools, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from xml.sax.saxutils import escape

st.set_page_config(page_title="ESTADÍSTICA PRO", page_icon="📊", layout="wide")

def parse_pop(txt):
    vals=[float(x.strip().replace(",", ".")) for x in txt.replace("\n"," ").replace(";"," ").split() if x.strip()]
    if len(vals)<2: raise ValueError("Ingresa al menos dos valores.")
    return vals

def pop_stats(p):
    return float(np.mean(p)), float(np.std(p, ddof=0))

def get_samples(p,n,repl):
    N=len(p)
    count=N**n if repl else (math.comb(N,n) if n<=N else 0)
    if count==0: raise ValueError("Sin reposición, n no puede superar N.")
    if count>200000: raise ValueError(f"Se generarían {count:,} muestras. Reduce n o la población (máximo 200.000).")
    ss=list(itertools.product(p,repeat=n)) if repl else list(itertools.combinations(p,n))
    return ss,np.array([np.mean(s) for s in ss],dtype=float)

def se_mean(p,n,repl):
    mu,s=pop_stats(p); se=s/math.sqrt(n)
    if not repl: se*=math.sqrt((len(p)-n)/(len(p)-1))
    return mu,s,se

def hist_png(vals,title,xlabel):
    fig,ax=plt.subplots(figsize=(8,3.4))
    ax.hist(vals,bins=min(15,max(5,len(np.unique(vals)))),edgecolor="black",alpha=.75)
    ax.set(title=title,xlabel=xlabel,ylabel="Frecuencia"); ax.grid(alpha=.2); fig.tight_layout()
    b=io.BytesIO(); fig.savefig(b,format="png",dpi=140,bbox_inches="tight"); plt.close(fig); b.seek(0); return b

def normal_png(mu,se,event,a,b):
    fig,ax=plt.subplots(figsize=(8,3.4))
    if se==0: ax.axvline(mu,label=f"Media={mu:.3f}")
    else:
        x=np.linspace(mu-4*se,mu+4*se,500); y=norm.pdf(x,mu,se); ax.plot(x,y)
        mask=(x<=a) if event=="Menor o igual que" else ((x>=a) if event=="Mayor o igual que" else ((x>=a)&(x<=b)))
        ax.fill_between(x[mask],y[mask],alpha=.3)
        ax.axvline(a,ls="--")
        if event=="Entre dos valores": ax.axvline(b,ls="--")
        ax.axvline(mu,ls=":",color="black",label=f"Media={mu:.3f}")
    ax.set(title="Aproximación normal de la distribución muestral",xlabel="Media muestral",ylabel="Densidad")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    out=io.BytesIO(); fig.savefig(out,format="png",dpi=140,bbox_inches="tight"); plt.close(fig); out.seek(0); return out

def pdf_report(title, sections):
    buf=io.BytesIO(); doc=SimpleDocTemplate(buf,pagesize=letter,rightMargin=1.7*cm,leftMargin=1.7*cm,topMargin=1.6*cm,bottomMargin=1.6*cm)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Cover",parent=styles["Title"],textColor=colors.HexColor("#176B55"),fontSize=20,leading=24,alignment=1,spaceAfter=12))
    styles.add(ParagraphStyle(name="Sec",parent=styles["Heading2"],textColor=colors.HexColor("#176B55"),fontSize=13,spaceBefore=12,spaceAfter=6))
    styles.add(ParagraphStyle(name="Txt",parent=styles["BodyText"],fontSize=9,leading=13,spaceAfter=5))
    story=[Paragraph(escape(title),styles["Cover"]),Paragraph("ESTADÍSTICA PRO | Informe académico",styles["Txt"]),Spacer(1,8)]
    for sec in sections:
        story.append(Paragraph(escape(sec["title"]),styles["Sec"]))
        for line in sec.get("text",[]): story.append(Paragraph(escape(line),styles["Txt"]))
        if sec.get("table"):
            data=[[str(v) for v in sec["table"][0]]]+[[str(v) for v in r] for r in sec["table"][1]]
            t=Table(data,repeatRows=1)
            t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#DCEFE8")),("GRID",(0,0),(-1,-1),.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),8),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F4F7F5")]),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
            story.extend([t,Spacer(1,8)])
        if sec.get("image"): story.extend([Image(sec["image"],width=16*cm,height=7*cm),Spacer(1,8)])
    def footer(canvas,doc):
        canvas.saveState(); canvas.setFont("Helvetica",8); canvas.setFillColor(colors.grey)
        canvas.drawString(1.7*cm,1*cm,"ESTADÍSTICA PRO | Informe académico")
        canvas.drawRightString(letter[0]-1.7*cm,1*cm,f"Página {doc.page}"); canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer); buf.seek(0); return buf.getvalue()

def sec_population(p):
    mu,s=pop_stats(p); rows=[[i+1,f"{x:g}",f"{x-mu:.4f}",f"{(x-mu)**2:.4f}"] for i,x in enumerate(p)]
    return {"title":"1. Población: media y desviación estándar","text":[f"Población: {p}. Tamaño N={len(p)}.",f"Media: mu = suma de los datos / N = {sum(p):.4f}/{len(p)} = {mu:.4f}.",f"Desviación poblacional: sigma = raíz de [suma (x-mu)^2/N] = {s:.4f}.","La desviación estándar mide qué tan dispersos están los valores respecto a la media."],"table":(["N.º","Valor x","x - mu","(x - mu)^2"],rows)}

def sec_means(p,n,repl):
    ss,ms=get_samples(p,n,repl); mu,s,se=se_mean(p,n,repl); name="con reposición" if repl else "sin reposición"
    rows=[[i+1," - ".join(f"{x:g}" for x in sample),f"{m:.4f}"] for i,(sample,m) in enumerate(zip(ss,ms))]
    formula=f"Error estándar = sigma/raíz(n) = {se:.4f}." if repl else f"Error estándar = [sigma/raíz(n)] × raíz[(N-n)/(N-1)] = {se:.4f}."
    return {"title":f"Distribución muestral de medias {name}","text":[f"Se enumeraron {len(ss)} muestras de tamaño n={n}, {name}.",f"Media de las medias = {np.mean(ms):.4f}; media poblacional = {mu:.4f}.",formula,f"Desviación estándar exacta de las medias = {np.std(ms):.4f}.","La desviación estándar de las medias se denomina error estándar."],"table":(["Muestra","Elementos","Media muestral"],rows),"image":hist_png(ms,"Distribución muestral de medias","Media muestral")}

def sec_probability(p,n,repl,event,a,b):
    mu,s,se=se_mean(p,n,repl)
    ss,ms=get_samples(p,n,repl)
    if event=="Menor o igual que":
        approx=float(norm.cdf(a,mu,se)) if se else float(mu<=a); exact=float(np.mean(ms<=a)); ev=f"X_bar <= {a:g}"
    elif event=="Mayor o igual que":
        approx=float(1-norm.cdf(a,mu,se)) if se else float(mu>=a); exact=float(np.mean(ms>=a)); ev=f"X_bar >= {a:g}"
    else:
        approx=float(norm.cdf(b,mu,se)-norm.cdf(a,mu,se)) if se else float(a<=mu<=b); exact=float(np.mean((ms>=a)&(ms<=b))); ev=f"{a:g} <= X_bar <= {b:g}"
    return {"title":"Probabilidad de eventos y distribución normal","text":[f"Evento: {ev}.",f"Media de la distribución de medias = {mu:.4f}; error estándar = {se:.4f}.",f"Probabilidad aproximada por normal = {approx:.6f} ({approx*100:.2f}%).",f"Probabilidad exacta por enumeración = {exact:.6f} ({exact*100:.2f}%).","La normal es una aproximación. Para poblaciones pequeñas y discretas, la enumeración exacta aporta la probabilidad del modelo muestral."],"image":normal_png(mu,se,event,a,b)}

def sec_props(p,K,n,repl):
    N=len(p); prop=K/N; ids=range(N)
    samples=list(itertools.product(ids,repeat=n)) if repl else list(itertools.combinations(ids,n))
    if len(samples)>200000: raise ValueError("Demasiadas muestras para el informe.")
    vals=np.array([sum(i<K for i in s)/n for s in samples]); u,c=np.unique(vals,return_counts=True)
    se=math.sqrt(prop*(1-prop)/n)
    if not repl: se*=math.sqrt((N-n)/(N-1))
    return {"title":"Distribución muestral de proporciones","text":[f"Éxitos K={K}; población N={N}; proporción poblacional p=K/N={prop:.4f}.",f"Tamaño muestral n={n}; muestreo {'con reposición' if repl else 'sin reposición'}.",f"Media exacta de las proporciones = {np.mean(vals):.4f}.",f"Desviación estándar exacta = {np.std(vals):.4f}; error estándar teórico = {se:.4f}.","La proporción muestral es la cantidad de éxitos de la muestra dividida entre n."],"table":(["Proporción muestral","Frecuencia","Probabilidad"],[[f"{v:.4f}",int(c0),f"{c0/len(vals):.4f}"] for v,c0 in zip(u,c)]),"image":hist_png(vals,"Distribución muestral de proporciones","Proporción muestral")}

def sec_sums(p1,p2,n1,n2,repl):
    _,m1=get_samples(p1,n1,repl); _,m2=get_samples(p2,n2,repl)
    if len(m1)*len(m2)>200000: raise ValueError("Demasiadas combinaciones; reduce los tamaños muestrales.")
    dif=np.array([x-y for x in m1 for y in m2]); sums=np.array([x+y for x in m1 for y in m2])
    mu1,_=pop_stats(p1); mu2,_=pop_stats(p2); _,_,e1=se_mean(p1,n1,repl); _,_,e2=se_mean(p2,n2,repl); sd=math.sqrt(e1**2+e2**2)
    return {"title":"Distribuciones muestrales de diferencias y sumas","text":[f"Población 1={p1}; n1={n1}; media mu1={mu1:.4f}.",f"Población 2={p2}; n2={n2}; media mu2={mu2:.4f}.","Se supone independencia entre las dos muestras.",f"Diferencia: media esperada mu1-mu2={mu1-mu2:.4f}; desviación teórica=raíz(SE1²+SE2²)={sd:.4f}. Resultado exacto: media={np.mean(dif):.4f}; desviación={np.std(dif):.4f}.",f"Suma: media esperada mu1+mu2={mu1+mu2:.4f}; desviación teórica={sd:.4f}. Resultado exacto: media={np.mean(sums):.4f}; desviación={np.std(sums):.4f}."],"table":(["Distribución","Media exacta","Desviación exacta"],[["Diferencia X_bar1-X_bar2",f"{np.mean(dif):.4f}",f"{np.std(dif):.4f}"],["Suma X_bar1+X_bar2",f"{np.mean(sums):.4f}",f"{np.std(sums):.4f}"]]),"image":hist_png(dif,"Distribución muestral de diferencias","X_bar1 - X_bar2")}

# -------------------- Interfaz principal --------------------
st.title("📊 ESTADÍSTICA PRO")
st.caption("Herramienta académica de distribuciones muestrales | Ingeniería Industrial")
st.write("Calcula medias, desviaciones, probabilidades, proporciones, diferencias y sumas. También genera informes académicos en PDF.")

with st.sidebar:
    st.header("Datos de entrada")
    raw=st.text_area("Población (separa los datos con comas, espacios o punto y coma)", "1, 9, 12, 14, 16, 20")
    try: population=parse_pop(raw); err=None
    except Exception as ex: population=[]; err=str(ex)
    if err: st.error(err)
    else: st.success(f"N = {len(population)}")
    n=st.number_input("Tamaño de muestra (n)",min_value=1,max_value=10,value=2,step=1)
if err: st.stop()
N=len(population); mu,sigma=pop_stats(population)
if n>N: st.warning("Para el muestreo sin reposición, n no puede superar N.")

st.subheader("Informes académicos en PDF")
st.write("Descarga un informe completo o un informe individual desde cada módulo.")
event=st.selectbox("Evento que se incluirá en el informe completo",["Menor o igual que","Mayor o igual que","Entre dos valores"])
a=st.number_input("Valor a",value=float(mu),key="all_a")
b=st.number_input("Valor b",value=float(mu+2),key="all_b")
if event!="Entre dos valores" or b>=a:
    try:
        sections=[sec_population(population)]
        for repl in (True,False):
            if n<=N: sections.append(sec_means(population,int(n),repl))
        for repl in (True,False):
            if n<=N: sections.append(sec_probability(population,int(n),repl,event,a,b))
        sections.append(sec_props(population,min(3,N),min(int(n),N),False))
        sections.append(sec_sums(population,[3,5,7,9,11,13],min(int(n),N),2,False))
        pdf=pdf_report("Informe estadístico completo",sections)
        st.download_button("📄 Descargar informe completo en PDF",pdf,"informe_estadistico_completo.pdf","application/pdf",type="primary")
    except Exception as ex: st.warning(f"No se pudo generar el informe completo: {ex}")
else: st.error("El valor b debe ser mayor o igual que a.")
st.divider()

t1,t2,t3,t4,t5=st.tabs(["1. Población y medias","2. Probabilidad y normal","3. Proporciones","4. Diferencias y sumas","5. Guía"])
with t1:
    st.subheader("Población y distribución muestral de medias")
    c1,c2,c3=st.columns(3); c1.metric("Media poblacional",f"{mu:.4f}"); c2.metric("Desviación poblacional",f"{sigma:.4f}"); c3.metric("N",N)
    repl_label=st.radio("Muestreo",["Con reposición","Sin reposición"],horizontal=True)
    repl=repl_label=="Con reposición"
    try:
        samples,means=get_samples(population,int(n),repl); _,_,se=se_mean(population,int(n),repl)
        st.metric("Cantidad de muestras",len(samples)); st.metric("Media de las medias",f"{np.mean(means):.4f}"); st.metric("Desviación exacta de las medias",f"{np.std(means):.4f}")
        st.write(f"Error estándar teórico: **{se:.4f}**")
        st.latex(r"\sigma_{\bar X}=\frac{\sigma}{\sqrt{n}}")
        if not repl: st.latex(r"\sigma_{\bar X}=\frac{\sigma}{\sqrt{n}}\sqrt{\frac{N-n}{N-1}}")
        df=pd.DataFrame({"Muestra":range(1,len(samples)+1),"Valores":[" - ".join(f"{x:g}" for x in s) for s in samples],"Media muestral":means})
        st.dataframe(df,use_container_width=True,hide_index=True)
        st.download_button("⬇️ Descargar tabla CSV",df.to_csv(index=False).encode("utf-8-sig"),"muestras.csv","text/csv")
        fig=hist_png(means,"Distribución muestral de medias","Media muestral"); st.image(fig); fig.seek(0)
        st.download_button("📄 Descargar informe de este módulo (PDF)",pdf_report("Informe: población y medias",[sec_population(population),sec_means(population,int(n),repl)]),"informe_poblacion_medias.pdf","application/pdf")
    except Exception as ex: st.error(str(ex))
with t2:
    st.subheader("Probabilidad de eventos y curva normal")
    rlab=st.radio("Muestreo",["Con reposición","Sin reposición"],horizontal=True,key="prob_repl"); repl=rlab=="Con reposición"
    ev=st.selectbox("Evento",["Menor o igual que","Mayor o igual que","Entre dos valores"],key="prob_event")
    aa=st.number_input("Valor a",value=float(mu),key="prob_a"); bb=st.number_input("Valor b",value=float(mu+2),key="prob_b")
    if ev!="Entre dos valores" or bb>=aa:
        try:
            sec=sec_probability(population,int(n),repl,ev,aa,bb)
            for line in sec["text"]: st.write(line)
            st.image(sec["image"])
            st.download_button("📄 Descargar informe de este módulo (PDF)",pdf_report("Informe: probabilidad y normal",[sec]),"informe_probabilidad.pdf","application/pdf")
        except Exception as ex: st.error(str(ex))
    else: st.error("El valor b debe ser mayor o igual que a.")
with t3:
    st.subheader("Distribución muestral de proporciones")
    K=st.number_input("Cantidad de éxitos K",min_value=0,max_value=N,value=min(3,N),step=1)
    np_=st.number_input("Tamaño muestral",min_value=1,max_value=N,value=min(int(n),N),step=1)
    rlab=st.radio("Muestreo",["Sin reposición","Con reposición"],horizontal=True,key="prop_repl"); repl=rlab=="Con reposición"
    try:
        sec=sec_props(population,int(K),int(np_),repl)
        for line in sec["text"]: st.write(line)
        st.table(pd.DataFrame(sec["table"][1],columns=sec["table"][0]))
        st.image(sec["image"])
        st.download_button("📄 Descargar informe de este módulo (PDF)",pdf_report("Informe: proporciones",[sec]),"informe_proporciones.pdf","application/pdf")
    except Exception as ex: st.error(str(ex))
with t4:
    st.subheader("Distribución muestral de diferencias y sumas")
    raw2=st.text_area("Segunda población","3, 5, 7, 9, 11, 13")
    try: p2=parse_pop(raw2); valid=True
    except Exception as ex: st.error(str(ex)); valid=False
    if valid:
        n1=st.number_input("Tamaño de muestra 1",min_value=1,max_value=min(N,8),value=min(int(n),N),key="n1")
        n2=st.number_input("Tamaño de muestra 2",min_value=1,max_value=min(len(p2),8),value=min(2,len(p2)),key="n2")
        rlab=st.radio("Muestreo para ambas poblaciones",["Sin reposición","Con reposición"],horizontal=True,key="sum_repl")
        try:
            sec=sec_sums(population,p2,int(n1),int(n2),rlab=="Con reposición")
            for line in sec["text"]: st.write(line)
            st.table(pd.DataFrame(sec["table"][1],columns=sec["table"][0]))
            st.image(sec["image"])
            st.download_button("📄 Descargar informe de este módulo (PDF)",pdf_report("Informe: diferencias y sumas",[sec]),"informe_diferencias_sumas.pdf","application/pdf")
        except Exception as ex: st.error(str(ex))
with t5:
    st.subheader("Guía de conceptos")
    with st.expander("Media poblacional"): st.write("Promedio de todos los valores: mu = suma de x / N.")
    with st.expander("Desviación estándar poblacional"): st.write("Mide la dispersión de los datos respecto a la media: sigma = raíz de [suma (x-mu)^2/N].")
    with st.expander("Distribución muestral de medias"): st.write("Distribución que se forma calculando la media de todas las muestras posibles de un mismo tamaño.")
    with st.expander("Con reposición y sin reposición"): st.write("Con reposición, un valor puede volver a seleccionarse. Sin reposición, no se repite dentro de una misma muestra.")
    with st.expander("Uso en Ingeniería Industrial"): st.write("El software automatiza cálculos, permite estudiar la variabilidad, analizar calidad y apoyar decisiones basadas en datos.")
    st.info("Los informes PDF incluyen fórmulas, sustitución numérica, resultados, tablas, gráficas e interpretaciones.")
