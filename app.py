import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import streamlit.components.v1 as components

# --- Page Configuration ---
st.set_page_config(page_title="Stacking Risk Prediction", page_icon="🏥", layout="wide")

# Helper function to render SHAP plots in Streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# --- Model Loading ---
@st.cache_resource
def load_model():
    try:
        # 确保 stacking_model.pkl 在同一目录下
        return joblib.load('stacking_model.pkl')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Sidebar: Clinical Input Data ---
st.sidebar.header("Patient Feature Input")

def user_input_features():
    # 1. Categorical Mappings (Match training labels)
    sp_options = {"Local Resection": 1, "Hemihepatectomy": 2, "Extended Hemihepatectomy": 3}
    uld_options = {"No": 0, "Yes": 1}
    cc_options = {"Grade A": 1, "Grade B": 2, "Grade C": 3}

    # Selection boxes
    sp_label = st.sidebar.selectbox('Surgical Procedure', list(sp_options.keys()))
    uld_label = st.sidebar.selectbox('With underlying liver disease', list(uld_options.keys()))
    cc_label = st.sidebar.selectbox('Child-Pugh Classification', list(cc_options.keys()))
    
    # Numerical inputs with specific steps
    tot = st.sidebar.number_input('Total Occlusion Time (min)', min_value=0.0, max_value=500.0, value=0.0, step=1.0)
    ibl = st.sidebar.number_input('Intraoperative Blood Loss (ml)', min_value=0.0, max_value=10000.0, value=0.0, step=50.0)
    ast = st.sidebar.number_input('AST (U/L)', min_value=0.0, max_value=5000.0, value=0.0, step=1.0)
    tbil = st.sidebar.number_input('TBIL (μmol/L)', min_value=0.0, max_value=1000.0, value=0.0, step=0.1)

    # Ensure keys match the training data column names exactly
    data = {
        'Surgical Procedure': sp_options[sp_label],
        'Total Occlusion Time': tot,
        'Intraoperative Blood Loss': ibl,
        'With underlying liver disease': uld_options[uld_label],
        'AST': ast,
        'TBIL': tbil,
        'Child-Pugh Classification': cc_options[cc_label]
    }
    return pd.DataFrame(data, index=[0]), {
        "Procedure": sp_label, "Disease": uld_label, "Class": cc_label
    }

input_df, display_labels = user_input_features()

# --- Main UI Interface ---
st.title("🏥 Clinical Risk Prediction System")
st.markdown("This system uses a Stacking model to assess patient risk.")

col_summary, col_prediction = st.columns([1, 2])

with col_summary:
    st.subheader("Patient Summary")
    # Table showing readable labels instead of encoded integers
    vis_df = input_df.copy()
    vis_df['Surgical Procedure'] = display_labels["Procedure"]
    vis_df['With underlying liver disease'] = display_labels["Disease"]
    vis_df['Child-Pugh Classification'] = display_labels["Class"]
    st.table(vis_df.T)

if st.button("Calculate Prediction", type="primary"):
    if model is not None:
        with st.spinner('Calculating risk and generating explanation...'):
            try:
               # 1. Prediction (预测)
                prediction_proba = model.predict_proba(input_df)
                risk_score = prediction_proba[0][1]  # 这一行定义了 risk_score
                
                THRESHOLD = 0.5  # 这一行定义了 THRESHOLD

                # --- 修复点开始 ---
                # 原代码错误地使用了 'prob' 和 'threshold'
                # 这里我们全部改用上面定义好的 'risk_score' 和 'THRESHOLD'
                risk_status = "High Risk" if risk_score >= THRESHOLD else "Low Risk"
                risk_color = "#E74C3C" if risk_status == "High Risk" else "#27AE60"

                # --- 2. Result Dashboard ---
                st.markdown("---")
                st.subheader("Assessment Result")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    # 这里把 prob 改为 risk_score
                    st.metric("Risk Probability", f"{risk_score:.4f}")
                    st.progress(float(risk_score))
                with res_col2:
                    st.markdown(f"Decision: <span style='color:{risk_color}; font-size:24px; font-weight:bold;'>{risk_status}</span>", unsafe_allow_html=True)
                    # 这里把 threshold 改为 THRESHOLD
                    st.caption(f"Decision Threshold: {THRESHOLD}")
                # --- 修复点结束 ---

               # --- 3. SHAP Interpretation ---
                st.markdown("---")
                st.subheader("Model Decision Interpretation (SHAP)")
                st.write("The chart below explains how each feature contributed to this specific prediction.")

                # Reference background data (Means derived from training set)
                bg_data = pd.DataFrame([[1.51, 24.45, 449.93, 0.18, 47.40, 39.97, 1.55]], 
                                       columns=input_df.columns)
                
                # Use KernelExplainer
                explainer = shap.KernelExplainer(model.predict_proba, bg_data)
                shap_values = explainer.shap_values(input_df)
                
                # Handling multi-output from shap_values
                if isinstance(shap_values, list):
                    # Class 1 (Risk) values
                    sv = shap_values[1]
                else:
                    sv = shap_values[..., 1] if shap_values.ndim == 3 else shap_values

                # Base value handling
                expected_val = explainer.expected_value
                if isinstance(expected_val, (list, np.ndarray)) and len(expected_val) > 1:
                    base_val = expected_val[1]
                else:
                    base_val = expected_val

                # --- 关键修改开始: 创建用于展示的 DataFrame ---
                # 复制一份输入数据
                display_df = input_df.copy()
                
                # 将分类变量的数字替换回文本 (利用你之前定义的 display_labels)
                # 注意：这里要确保类型转换，不然 numpy 数组可能会因为混合类型报错，最好转为 object 或 string
                display_df = display_df.astype(object) 
                display_df['Surgical Procedure'] = display_labels["Procedure"]
                display_df['With underlying liver disease'] = display_labels["Disease"]
                display_df['Child-Pugh Classification'] = display_labels["Class"]
                # --- 关键修改结束 ---

                # Render Force Plot
                p = shap.force_plot(
                    float(base_val), 
                    sv.flatten(), 
                    display_df.iloc[0].values,  # <--- 修改这里：传入包含文本的 display_df
                    feature_names=input_df.columns.tolist(),
                    link='logit' 
                )
                st_shap(p, height=180)
                st.caption("Interpretation: Red arrows push the risk higher; blue arrows push it lower.")

            except Exception as e:
                st.error(f"Analysis failed: {e}")
    else:
        st.warning("Prediction unavailable: Model not loaded correctly.")