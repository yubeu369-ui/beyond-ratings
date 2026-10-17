from pathlib import Path
import json, re
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf

SITE = "https://yubeu369-ui.github.io/beyond-ratings/"
LOCAL = Path("data/matched302.csv")

def load_frozen_sample():
    if LOCAL.exists():
        df = pd.read_csv(LOCAL)
        # Accept either the original packaged field names or the presentation-friendly names.
        df = df.rename(columns={
            "vote_average":"Rating", "popularity":"Popularity",
            "release_month":"ReleaseMonth"
        })
    else:
        import requests
        html = requests.get(SITE, timeout=30).text
        m = re.search(r"const D=(\{.*?\}), N=D\.title\.length;", html, re.S)
        if not m:
            raise RuntimeError("Could not locate the embedded frozen sample.")
        d = json.loads(m.group(1))
        df = pd.DataFrame(d).rename(columns={
            "title":"Movie","year":"Year","genre":"Genre","release_month":"ReleaseMonth",
            "season":"Season","vote_average":"Rating","popularity":"Popularity",
            "opening_theaters":"TheatersOpenWeek","budget_m":"Budget",
            "domestic_gross_m":"DomesticGross","opening_weekend_m":"OpeningWeekend",
            "post_opening_gross_m":"LaterRevenue"
        })
    assert len(df) == 302
    return df

def fisher_ci(r,n,alpha=.05):
    z=np.arctanh(r); se=1/np.sqrt(n-3); zc=stats.norm.ppf(1-alpha/2)
    return np.tanh([z-zc*se,z+zc*se])

def williams_test(r12,r13,r23,n):
    k=1-r12**2-r13**2-r23**2+2*r12*r13*r23
    num=(r12-r13)*np.sqrt((n-1)*(1+r23))
    den=np.sqrt(2*k*(n-1)/(n-3)+((r12+r13)**2/4)*(1-r23)**3)
    t=num/den
    return t,n-3,stats.t.sf(t,n-3)

df=load_frozen_sample()
print(f"Frozen analytic sample: n={len(df)}, years {df.Year.min()}-{df.Year.max()}")

corr={}
for label,col in [("Rating","Rating"),("Popularity","Popularity"),("Opening theaters","TheatersOpenWeek")]:
    r,p=stats.pearsonr(df[col],df["DomesticGross"])
    corr[label]=(r,fisher_ci(r,len(df)),p)
    print(f"{label}: r={r:.3f}, 95% CI [{corr[label][1][0]:.3f}, {corr[label][1][1]:.3f}], p={p:.3g}")

r_rating=corr["Rating"][0]; r_pop=corr["Popularity"][0]; r_th=corr["Opening theaters"][0]
wp=williams_test(r_pop,r_rating,stats.pearsonr(df.Popularity,df.Rating)[0],len(df))
wt=williams_test(r_th,r_rating,stats.pearsonr(df.TheatersOpenWeek,df.Rating)[0],len(df))
holm=multipletests([wp[2],wt[2]],method="holm")[1]
print(f"Popularity - Rating: delta r={r_pop-r_rating:+.4f}, t({wp[1]})={wp[0]:.4f}, Holm p={holm[0]:.4f}")
print(f"Theaters - Rating: delta r={r_th-r_rating:+.4f}, t({wt[1]})={wt[0]:.4f}, Holm p={holm[1]:.4f}")

for col in ["Rating","Popularity","TheatersOpenWeek","Budget","DomesticGross"]:
    df["z_"+col]=(df[col]-df[col].mean())/df[col].std(ddof=0)
rq2=smf.ols("z_DomesticGross ~ z_Rating + z_Popularity + z_TheatersOpenWeek + z_Budget + C(Genre) + C(ReleaseMonth)",data=df).fit()
hc3=rq2.get_robustcov_results(cov_type="HC3")
names=rq2.model.exog_names
print(f"RQ2 fit: R2={rq2.rsquared:.3f}, adjusted R2={rq2.rsquared_adj:.3f}, F={rq2.fvalue:.2f}, p={rq2.f_pvalue:.3g}")
for nm in ["z_Rating","z_Popularity","z_TheatersOpenWeek"]:
    i=names.index(nm); ci=hc3.conf_int()[i]
    print(f"{nm}: beta={rq2.params[nm]:.3f}, HC3 95% CI [{ci[0]:.3f}, {ci[1]:.3f}], p={hc3.pvalues[i]:.4g}")

def contrast(a,b):
    c=np.zeros(len(names)); c[names.index(a)]=1; c[names.index(b)]=-1
    return rq2.t_test(c)
print("Popularity - Rating contrast:",contrast("z_Popularity","z_Rating"))
print("Theaters - Rating contrast:",contrast("z_TheatersOpenWeek","z_Rating"))

for outcome in ["OpeningWeekend","LaterRevenue","DomesticGross"]:
    m=smf.ols(f"{outcome} ~ TheatersOpenWeek + Budget + C(Genre) + C(ReleaseMonth)",data=df).fit()
    h=m.get_robustcov_results(cov_type="HC3")
    i=m.model.exog_names.index("TheatersOpenWeek"); ci=h.conf_int()[i]; b=m.params["TheatersOpenWeek"]
    print(f"{outcome}: +100 theaters = +{100*b:.2f} million USD, HC3 95% CI [{100*ci[0]:.2f}, {100*ci[1]:.2f}], p={h.pvalues[i]:.4g}, R2={m.rsquared:.3f}")

base=smf.ols("DomesticGross ~ TheatersOpenWeek + Budget + C(Genre) + C(ReleaseMonth)",data=df).fit()
bp=het_breuschpagan(base.resid,base.model.exog)
print(f"Breusch-Pagan F p={bp[3]:.4f}; residual skew={stats.skew(base.resid,bias=False):.3f}; JB p={stats.jarque_bera(base.resid).pvalue:.3g}")
X=pd.DataFrame(base.model.exog,columns=base.model.exog_names)
vifs=[(c,variance_inflation_factor(X.values,i)) for i,c in enumerate(X.columns) if c!="Intercept"]
print("Max VIF:",max(vifs,key=lambda x:x[1]))
print("Cook D > 4/n:",int((base.get_influence().cooks_distance[0]>4/len(df)).sum()))
df["log1p_DomesticGross"]=np.log1p(df.DomesticGross)
logm=smf.ols("log1p_DomesticGross ~ TheatersOpenWeek + Budget + C(Genre) + C(ReleaseMonth)",data=df).fit().get_robustcov_results(cov_type="HC3")
i=logm.model.exog_names.index("TheatersOpenWeek"); b=logm.params[i]
print(f"Log outcome: +100 theaters approximately {(np.exp(100*b)-1)*100:.2f}% gross, p={logm.pvalues[i]:.3g}")
