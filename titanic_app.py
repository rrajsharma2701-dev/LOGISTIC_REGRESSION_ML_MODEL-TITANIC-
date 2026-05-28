import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="TITANIC — Fate Engine",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── MODEL ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def build_model():
    df = sns.load_dataset('titanic')
    df.drop(['alive','embarked','who','deck','class','adult_male'], axis=1, inplace=True)
    df['age'] = df['age'].fillna(df['age'].median())
    df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])
    df_enc = pd.get_dummies(df, columns=['sex','embark_town'], drop_first=True)
    X = df_enc.drop('survived', axis=1)
    y = df_enc['survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train)
    Xte = sc.transform(X_test)
    m = LogisticRegression(max_iter=500)
    m.fit(Xtr, y_train)
    yp = m.predict(Xte)
    acc = accuracy_score(y_test, yp)
    cm  = confusion_matrix(y_test, yp)
    cr  = classification_report(y_test, yp, output_dict=True)
    return m, sc, list(X.columns), acc, cm, cr, df_enc

model, scaler, COLS, ACC, CM, CR, DF = build_model()

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,600;0,700;1,300;1,600&family=Cinzel:wght@400;600;900&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main { background: #04080f !important; }

[data-testid="stHeader"]        { background: transparent !important; }
[data-testid="stSidebar"]       { background: #060c18 !important; }
[data-testid="stMainBlockContainer"] { padding-top: 0 !important; }

/* ── HERO ── */
.hero-wrap {
    width: 100%;
    min-height: 340px;
    background:
        radial-gradient(ellipse 120% 60% at 50% -10%, rgba(180,140,50,0.18) 0%, transparent 65%),
        radial-gradient(ellipse 80% 40% at 20% 110%, rgba(10,30,80,0.9) 0%, transparent 60%),
        linear-gradient(180deg, #04080f 0%, #070e1f 60%, #050a16 100%);
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3.5rem 2rem 2.5rem;
    overflow: hidden;
}

/* ocean wave lines */
.hero-wrap::before {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 60px;
    background: repeating-linear-gradient(
        90deg,
        transparent 0px,
        transparent 18px,
        rgba(100,160,220,0.06) 18px,
        rgba(100,160,220,0.06) 36px
    );
    mask-image: linear-gradient(0deg, black 0%, transparent 100%);
}

.hero-stamp {
    font-family: 'Cinzel', serif;
    font-size: 0.68rem;
    letter-spacing: 0.55em;
    color: #b8960a;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

.hero-ship-line {
    width: 260px;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, #c9a430 30%, #e8cc6a 50%, #c9a430 70%, transparent 100%);
    margin: 0.4rem auto;
}

.hero-title {
    font-family: 'Cinzel', serif;
    font-size: clamp(3rem, 8vw, 6.5rem);
    font-weight: 900;
    color: #f0e4b8;
    letter-spacing: 0.18em;
    line-height: 1;
    text-shadow:
        0 0 40px rgba(200,160,50,0.5),
        0 0 80px rgba(200,160,50,0.2),
        0 2px 0 rgba(0,0,0,0.8);
    position: relative;
}

/* 3‑D bevel on each letter */
.hero-title span {
    display: inline-block;
    transform: perspective(300px) rotateX(8deg);
    transform-origin: center bottom;
}

.hero-sub {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 1.05rem;
    color: #8ab4d4;
    letter-spacing: 0.25em;
    margin-top: 0.6rem;
}

.hero-anchor {
    font-size: 1.4rem;
    margin: 0.6rem 0;
    opacity: 0.5;
}

/* ── METRIC STRIP ── */
.metric-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: rgba(200,164,50,0.15);
    border-top: 1px solid rgba(200,164,50,0.25);
    border-bottom: 1px solid rgba(200,164,50,0.25);
    margin-bottom: 2rem;
}
.metric-cell {
    background: #04080f;
    padding: 1rem;
    text-align: center;
    transition: background 0.3s;
}
.metric-cell:hover { background: #090f1f; }
.metric-num {
    font-family: 'Cinzel', serif;
    font-size: 2rem;
    font-weight: 600;
    color: #c9a430;
    line-height: 1;
}
.metric-lbl {
    font-family: 'Cormorant Garamond', serif;
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    color: #5a8aaa;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* ── SECTION HEADING ── */
.sec-head {
    font-family: 'Cinzel', serif;
    font-size: 0.75rem;
    letter-spacing: 0.35em;
    color: #c9a430;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(200,164,50,0.2);
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
}

/* ── FORM PANEL ── */
.form-panel {
    background: linear-gradient(145deg, #060d1c 0%, #0a1428 100%);
    border: 1px solid rgba(200,164,50,0.18);
    padding: 1.8rem;
    position: relative;
}
.form-panel::before {
    content: '⚓';
    position: absolute;
    top: 0.8rem; right: 1rem;
    font-size: 5rem;
    opacity: 0.04;
    line-height: 1;
}

/* ── RESULT PANEL ── */
.verdict-survived {
    background: linear-gradient(145deg, #021510 0%, #04201a 100%);
    border: 1.5px solid #1e7a50;
    padding: 2.2rem 1.8rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.verdict-survived::after {
    content: '🛟';
    position: absolute;
    bottom: -10px; right: 5px;
    font-size: 6rem;
    opacity: 0.07;
}
.verdict-perished {
    background: linear-gradient(145deg, #150202 0%, #200606 100%);
    border: 1.5px solid #7a1e1e;
    padding: 2.2rem 1.8rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.verdict-perished::after {
    content: '⚓';
    position: absolute;
    bottom: -10px; right: 5px;
    font-size: 6rem;
    opacity: 0.07;
}

.verdict-label {
    font-family: 'Cinzel', serif;
    font-size: 2.8rem;
    font-weight: 900;
    letter-spacing: 0.2em;
    line-height: 1;
    text-shadow: 0 0 30px currentColor;
}
.survived-color  { color: #2ecc71; }
.perished-color  { color: #e74c3c; }
.verdict-flavour {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 1rem;
    margin-top: 0.5rem;
    opacity: 0.75;
}

/* probability bar */
.prob-track {
    background: rgba(255,255,255,0.07);
    height: 6px;
    margin: 1rem 0 0.3rem;
    border-radius: 0;
}
.prob-fill-s { background: linear-gradient(90deg,#1a7a40,#2ecc71); height: 6px; }
.prob-fill-p { background: linear-gradient(90deg,#7a1a1a,#e74c3c); height: 6px; }
.prob-pct {
    font-family: 'Cinzel', serif;
    font-size: 2rem;
    font-weight: 600;
}

/* ticket */
.ticket {
    background: #07101f;
    border: 1px solid rgba(200,164,50,0.2);
    padding: 1.2rem 1.4rem;
    margin-top: 1.2rem;
}
.trow {
    display: flex;
    justify-content: space-between;
    padding: 0.35rem 0;
    border-bottom: 1px solid rgba(200,164,50,0.07);
    font-family: 'Cormorant Garamond', serif;
    font-size: 1rem;
}
.tlabel { color: #4a7a9a; font-style: italic; }
.tval   { color: #d4b84a; }

/* ── IDLE PANEL ── */
.idle-panel {
    background: #060d1c;
    border: 1px solid rgba(200,164,50,0.12);
    padding: 3.5rem 1.5rem;
    text-align: center;
}
.idle-quote {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 1.15rem;
    color: #3a6a8a;
    line-height: 1.7;
    max-width: 320px;
    margin: 0 auto;
}

/* ── WIDGET OVERRIDES ── */
label,
[data-testid="stWidgetLabel"] p {
    color: #5a8aaa !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-style: italic !important;
    font-size: 1rem !important;
    letter-spacing: 0.05em !important;
}
input, textarea,
div[data-baseweb="select"] > div,
div[data-baseweb="input"]  > div > input {
    background: #07101f !important;
    border: 1px solid rgba(200,164,50,0.25) !important;
    color: #e0d4a0 !important;
    border-radius: 0 !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1rem !important;
}
[data-baseweb="popover"] [role="listbox"] {
    background: #07101f !important;
    border: 1px solid rgba(200,164,50,0.25) !important;
}
[data-baseweb="option"]:hover { background: #0d1c30 !important; }
[data-baseweb="option"] span  { color: #e0d4a0 !important; font-family:'Cormorant Garamond',serif!important; }

.stSlider > div > div { background: rgba(200,164,50,0.15) !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #c9a430 !important; border-color: #c9a430 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7a5e08, #c9a430, #e8cc6a) !important;
    color: #04080f !important;
    font-family: 'Cinzel', serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.3em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.9rem 2rem !important;
    width: 100%;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 24px rgba(200,164,50,0.2) !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 32px rgba(200,164,50,0.45) !important;
    transform: translateY(-2px) !important;
}

/* analytics section */
.analytics-wrap {
    background: #060d1c;
    border: 1px solid rgba(200,164,50,0.14);
    padding: 1.8rem;
    margin-top: 0.5rem;
}

footer, #MainMenu { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-stamp">R.M.S. White Star Line · North Atlantic · April 14–15, 1912</div>
  <div class="hero-ship-line"></div>
  <div class="hero-title"><span>TITANIC</span></div>
  <div class="hero-ship-line"></div>
  <div class="hero-sub">Survival Prediction Engine &nbsp;·&nbsp; Logistic Regression &nbsp;·&nbsp; Machine Learning</div>
  <div class="hero-anchor">⚓</div>
</div>
""", unsafe_allow_html=True)

# ── METRIC STRIP ──────────────────────────────────────────────────────────────
survived_n = int(DF['survived'].sum())
total_n    = len(DF)
perished_n = total_n - survived_n

st.markdown(f"""
<div class="metric-strip">
  <div class="metric-cell">
    <div class="metric-num">{total_n}</div>
    <div class="metric-lbl">Passengers</div>
  </div>
  <div class="metric-cell">
    <div class="metric-num">{survived_n}</div>
    <div class="metric-lbl">Survived</div>
  </div>
  <div class="metric-cell">
    <div class="metric-num">{perished_n}</div>
    <div class="metric-lbl">Perished</div>
  </div>
  <div class="metric-cell">
    <div class="metric-num">{ACC*100:.1f}%</div>
    <div class="metric-lbl">Model Accuracy</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── MAIN LAYOUT ───────────────────────────────────────────────────────────────
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="sec-head">⚓ &nbsp; Passenger Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="form-panel">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        pclass = st.selectbox("Passenger Class",
            options=[1,2,3],
            format_func=lambda x:{1:"1st — First Class",2:"2nd — Second Class",3:"3rd — Steerage"}[x])
        sex = st.selectbox("Sex", ["male","female"])
        age = st.slider("Age (years)", 1, 80, 28)
        sibsp = st.selectbox("Siblings / Spouse aboard", list(range(6)))
    with c2:
        parch = st.selectbox("Parents / Children aboard", list(range(7)))
        fare  = st.number_input("Ticket Fare (£)", 0.0, 600.0, 32.2, 0.5)
        alone = st.selectbox("Travelling alone?", ["Yes","No"])
        embark = st.selectbox("Port of Embarkation",
            ["Southampton","Cherbourg","Queenstown"])

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⚓  DETERMINE FATE"):
        alone_val = 1 if alone=="Yes" else 0
        sex_male  = 1 if sex=="male"  else 0
        qt = 1 if embark=="Queenstown"  else 0
        sn = 1 if embark=="Southampton" else 0
        row = pd.DataFrame(
            [[pclass,age,sibsp,parch,fare,alone_val,sex_male,qt,sn]],
            columns=COLS)
        row_sc  = scaler.transform(row)
        pred    = model.predict(row_sc)[0]
        prob    = model.predict_proba(row_sc)[0]
        st.session_state['res'] = dict(
            pred=pred, prob_s=prob[1], prob_p=prob[0],
            pclass=pclass, sex=sex, age=age,
            fare=fare, embark=embark, sibsp=sibsp,
            parch=parch, alone=alone)


with right:
    st.markdown('<div class="sec-head">🔮 &nbsp; Fate of the Passenger</div>', unsafe_allow_html=True)

    if 'res' not in st.session_state:
        st.markdown("""
        <div class="idle-panel">
          <div style="font-size:3.5rem; margin-bottom:1.2rem; opacity:0.3;">🚢</div>
          <div class="idle-quote">
            "She was the largest ship afloat and deemed unsinkable.<br>
            On her maiden voyage, she met the iceberg."
          </div>
          <div style="margin-top:1.5rem; font-family:'Cinzel',serif;
               font-size:0.65rem; letter-spacing:0.3em; color:#1a3a5a;">
            ENTER PASSENGER DETAILS &amp; PREDICT
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        r = st.session_state['res']
        s = r['pred'] == 1
        pct = r['prob_s'] * 100

        if s:
            st.markdown(f"""
            <div class="verdict-survived">
              <div class="verdict-label survived-color">SURVIVED</div>
              <div class="verdict-flavour" style="color:#6fcf97;">
                This passenger reaches the lifeboats safely.
              </div>
              <div class="prob-track">
                <div class="prob-fill-s" style="width:{pct:.0f}%;"></div>
              </div>
              <div style="display:flex; justify-content:space-between;
                   font-family:'Cormorant Garamond',serif; font-size:0.8rem;
                   color:#2a6a44; letter-spacing:0.15em; text-transform:uppercase;">
                <span>Survival Probability</span>
                <span class="prob-pct survived-color">{pct:.1f}%</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-perished">
              <div class="verdict-label perished-color">PERISHED</div>
              <div class="verdict-flavour" style="color:#e07070;">
                This passenger does not survive the night.
              </div>
              <div class="prob-track">
                <div class="prob-fill-p" style="width:{pct:.0f}%;"></div>
              </div>
              <div style="display:flex; justify-content:space-between;
                   font-family:'Cormorant Garamond',serif; font-size:0.8rem;
                   color:#6a2a2a; letter-spacing:0.15em; text-transform:uppercase;">
                <span>Survival Probability</span>
                <span class="prob-pct perished-color">{pct:.1f}%</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        class_map = {1:"First Class",2:"Second Class",3:"Third Class / Steerage"}
        st.markdown(f"""
        <div class="ticket">
          <div style="font-family:'Cinzel',serif; font-size:0.65rem;
               letter-spacing:0.3em; color:#5a3a08; margin-bottom:0.7rem;">
               PASSENGER TICKET — R.M.S. TITANIC
          </div>
          <div class="trow"><span class="tlabel">Class</span>
               <span class="tval">{class_map[r['pclass']]}</span></div>
          <div class="trow"><span class="tlabel">Sex</span>
               <span class="tval">{r['sex'].capitalize()}</span></div>
          <div class="trow"><span class="tlabel">Age</span>
               <span class="tval">{r['age']} years</span></div>
          <div class="trow"><span class="tlabel">Siblings / Spouse</span>
               <span class="tval">{r['sibsp']}</span></div>
          <div class="trow"><span class="tlabel">Parents / Children</span>
               <span class="tval">{r['parch']}</span></div>
          <div class="trow"><span class="tlabel">Fare Paid</span>
               <span class="tval">£{r['fare']:.2f}</span></div>
          <div class="trow"><span class="tlabel">Travelling Alone</span>
               <span class="tval">{r['alone']}</span></div>
          <div class="trow" style="border:none;">
               <span class="tlabel">Embarked</span>
               <span class="tval">{r['embark']}</span></div>
        </div>
        """, unsafe_allow_html=True)


# ── MODEL ANALYTICS ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="sec-head">📊 &nbsp; Model Analytics</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["  Confusion Matrix  ", "  Classification Report  ", "  Survival Insights  "])

with tab1:
    col_cm, col_cr = st.columns([1,1.4])
    with col_cm:
        fig, ax = plt.subplots(figsize=(4,3.5))
        fig.patch.set_facecolor('#06101e')
        ax.set_facecolor('#06101e')
        sns.heatmap(CM, annot=True, fmt='d', cmap='YlOrBr',
                    ax=ax, linewidths=0.5, linecolor='#0a1428',
                    annot_kws={'size':16,'color':'white','weight':'bold'})
        ax.set_xlabel('Predicted', color='#5a8aaa', fontsize=9, labelpad=8)
        ax.set_ylabel('Actual',    color='#5a8aaa', fontsize=9, labelpad=8)
        ax.set_title('Confusion Matrix', color='#c9a430', fontsize=10,
                     fontfamily='serif', pad=10)
        ax.tick_params(colors='#5a8aaa', labelsize=9)
        for spine in ax.spines.values(): spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with col_cr:
        st.markdown(f"""
        <div style="padding:1rem 0;">
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:12px;">
          <div style="background:#07101f; border:1px solid rgba(200,164,50,0.18);
               padding:1rem; text-align:center;">
            <div style="font-family:'Cinzel',serif; font-size:1.6rem; color:#c9a430;">
              {ACC*100:.1f}%</div>
            <div style="font-size:0.7rem; letter-spacing:0.2em; color:#5a8aaa;
                 text-transform:uppercase; margin-top:0.3rem;">Accuracy</div>
          </div>
          <div style="background:#07101f; border:1px solid rgba(200,164,50,0.18);
               padding:1rem; text-align:center;">
            <div style="font-family:'Cinzel',serif; font-size:1.6rem; color:#2ecc71;">
              {float(CR['1']['recall'])*100:.1f}%</div>
            <div style="font-size:0.7rem; letter-spacing:0.2em; color:#5a8aaa;
                 text-transform:uppercase; margin-top:0.3rem;">Recall (Survived)</div>
          </div>
          <div style="background:#07101f; border:1px solid rgba(200,164,50,0.18);
               padding:1rem; text-align:center;">
            <div style="font-family:'Cinzel',serif; font-size:1.6rem; color:#e74c3c;">
              {float(CR['0']['precision'])*100:.1f}%</div>
            <div style="font-size:0.7rem; letter-spacing:0.2em; color:#5a8aaa;
                 text-transform:uppercase; margin-top:0.3rem;">Precision (Perished)</div>
          </div>
          <div style="background:#07101f; border:1px solid rgba(200,164,50,0.18);
               padding:1rem; text-align:center;">
            <div style="font-family:'Cinzel',serif; font-size:1.6rem; color:#8ab4d4;">
              {float(CR['1']['f1-score'])*100:.1f}%</div>
            <div style="font-size:0.7rem; letter-spacing:0.2em; color:#5a8aaa;
                 text-transform:uppercase; margin-top:0.3rem;">F1-Score (Survived)</div>
          </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    cr_df = pd.DataFrame({
        'Class':     ['Perished (0)','Survived (1)','Macro Avg','Weighted Avg'],
        'Precision': [f"{float(CR['0']['precision']):.2f}",
                      f"{float(CR['1']['precision']):.2f}",
                      f"{float(CR['macro avg']['precision']):.2f}",
                      f"{float(CR['weighted avg']['precision']):.2f}"],
        'Recall':    [f"{float(CR['0']['recall']):.2f}",
                      f"{float(CR['1']['recall']):.2f}",
                      f"{float(CR['macro avg']['recall']):.2f}",
                      f"{float(CR['weighted avg']['recall']):.2f}"],
        'F1-Score':  [f"{float(CR['0']['f1-score']):.2f}",
                      f"{float(CR['1']['f1-score']):.2f}",
                      f"{float(CR['macro avg']['f1-score']):.2f}",
                      f"{float(CR['weighted avg']['f1-score']):.2f}"],
        'Support':   [str(int(CR['0']['support'])),
                      str(int(CR['1']['support'])),
                      str(int(CR['macro avg']['support'])),
                      str(int(CR['weighted avg']['support']))],
    })
    st.dataframe(
        cr_df.style
            .set_properties(**{
                'background-color':'#06101e',
                'color':'#d4b84a',
                'border':'1px solid #0d1c30',
                'font-family':'Cormorant Garamond, serif',
                'font-size':'14px'
            })
            .set_table_styles([{
                'selector':'th',
                'props':[('background','#0a1428'),
                         ('color','#8ab4d4'),
                         ('font-family','Cinzel, serif'),
                         ('letter-spacing','0.15em'),
                         ('font-size','11px')]
            }]),
        use_container_width=True, hide_index=True
    )

with tab3:
    ins1, ins2 = st.columns(2)
    with ins1:
        fig2, ax2 = plt.subplots(figsize=(5,3.5))
        fig2.patch.set_facecolor('#06101e')
        ax2.set_facecolor('#06101e')
        surv_by_sex = DF.groupby('sex_male')['survived'].mean()
        bars = ax2.bar(['Female','Male'], surv_by_sex.values,
                       color=['#c9a430','#2a6aaa'], width=0.5,
                       edgecolor='none')
        for bar, val in zip(bars, surv_by_sex.values):
            ax2.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.01, f'{val*100:.1f}%',
                     ha='center', va='bottom', color='#d4b84a',
                     fontsize=10, fontfamily='serif')
        ax2.set_title('Survival Rate by Sex', color='#c9a430',
                      fontsize=10, fontfamily='serif', pad=10)
        ax2.set_ylabel('Survival Rate', color='#5a8aaa', fontsize=9)
        ax2.tick_params(colors='#5a8aaa', labelsize=9)
        ax2.set_ylim(0,1)
        for spine in ax2.spines.values(): spine.set_color('#0d1c30')
        ax2.yaxis.grid(True, color='#0d1c30', linestyle='--', alpha=0.5)
        ax2.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
    with ins2:
        fig3, ax3 = plt.subplots(figsize=(5,3.5))
        fig3.patch.set_facecolor('#06101e')
        ax3.set_facecolor('#06101e')
        surv_by_cls = DF.groupby('pclass')['survived'].mean()
        bars2 = ax3.bar([f'Class {c}' for c in surv_by_cls.index],
                        surv_by_cls.values,
                        color=['#c9a430','#5a8aaa','#8b6914'],
                        width=0.5, edgecolor='none')
        for bar, val in zip(bars2, surv_by_cls.values):
            ax3.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.01, f'{val*100:.1f}%',
                     ha='center', va='bottom', color='#d4b84a',
                     fontsize=10, fontfamily='serif')
        ax3.set_title('Survival Rate by Class', color='#c9a430',
                      fontsize=10, fontfamily='serif', pad=10)
        ax3.set_ylabel('Survival Rate', color='#5a8aaa', fontsize=9)
        ax3.tick_params(colors='#5a8aaa', labelsize=9)
        ax3.set_ylim(0,1)
        for spine in ax3.spines.values(): spine.set_color('#0d1c30')
        ax3.yaxis.grid(True, color='#0d1c30', linestyle='--', alpha=0.5)
        ax3.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2.5rem 0 1.5rem;
     border-top: 1px solid rgba(200,164,50,0.1); margin-top:2rem;">
  <div style="font-family:'Cinzel',serif; font-size:0.65rem;
       letter-spacing:0.4em; color:#1a3a5a; margin-bottom:0.6rem;">
    ⚓ &nbsp; IN MEMORY OF THE 1,517 SOULS LOST ON APRIL 15, 1912 &nbsp; ⚓
  </div>
  <div style="font-family:'Cormorant Garamond',serif; font-style:italic;
       font-size:0.85rem; color:#1a3a5a; letter-spacing:0.1em;">
    Built with Logistic Regression · Scikit-learn · Streamlit
  </div>
</div>
""", unsafe_allow_html=True)