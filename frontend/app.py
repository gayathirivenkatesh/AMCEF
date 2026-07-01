import streamlit as st
import requests
import json
import plotly.graph_objects as go
import numpy as np
from streamlit_mic_recorder import mic_recorder
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AMCEF Emotion Intelligence Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ PREMIUM STYLE ------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #111419;
    color: #e6f1f3;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #1a1e26;
    border-right: 1px solid #2d333b;
}
.card {
    background: #1b2129;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid #2d333b;
    margin-bottom: 24px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.card-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.9rem;
    color: #00e6c3;
    font-weight: 500;
}
.status-dot {
    height: 10px;
    width: 10px;
    background-color: #00e6c3;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 8px #00e6c3;
}
.header-container {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding-bottom: 20px;
    border-bottom: 1px solid #2d333b;
    margin-bottom: 30px;
}

/* Timeline */
.timeline {
    position: relative;
    padding-left: 20px;
    margin-top: 20px;
}
.timeline::before {
    content: '';
    position: absolute;
    left: 4px;
    top: 0;
    bottom: 0;
    width: 2px;
    background: #2d333b;
}
.timeline-item {
    position: relative;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    background: #222933;
    padding: 12px 16px;
    border-radius: 8px;
    border: 1px solid #2d333b;
}
.timeline-item::before {
    content: '';
    position: absolute;
    left: -21px;
    top: 50%;
    transform: translateY(-50%);
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #8b949e;
    border: 2px solid #111419;
}
.timeline-item.final {
    border-color: #00e6c3;
    background: rgba(0, 230, 195, 0.05);
}
.timeline-item.final::before {
    background: #00e6c3;
    box-shadow: 0 0 8px #00e6c3;
}
.timeline-item span:first-child {
    font-weight: 500;
    color: #c9d1d9;
}
.timeline-item span:last-child {
    font-family: monospace;
    color: #8b949e;
    font-size: 1.1em;
}

/* Button override */
.stButton > button {
    height: 54px;
    font-weight: bold;
    font-size: 1.1rem;
    border-radius: 12px;
    background-color: #00e6c3;
    color: #111419;
    border: none;
}
.stButton > button:hover {
    background-color: #00ffda;
    color: #111419;
    box-shadow: 0 0 10px rgba(0, 230, 195, 0.4);
}
</style>
""", unsafe_allow_html=True)

# ------------------ HELPERS ------------------
emotion_styles = {
    "HAPPY": ("#f1c40f", "😄"),  # Yellow
    "SAD": ("#3498db", "😢"),    # Blue
    "SADNESS": ("#3498db", "😢"),# Blue
    "ANGRY": ("#e74c3c", "😠"),  # Red
    "FEAR": ("#9b59b6", "😨"),   # Purple
    "SURPRISE": ("#e67e22", "😲"),
    "NEUTRAL": ("#95a5a6", "😐"), # Gray
}

def get_emotion_style(emotion):
    return emotion_styles.get(emotion.upper(), ("#95a5a6", "🎭"))

    with col_r:
        # --- RIGHT PANEL ---
        col_audio, col_video = st.columns(2)
        
        with col_audio:
            a_conf = audio.get("confidence", 0.0)
            a_col, a_emj = get_emotion_style(audio.get("emotion", "N/A"))
            
            st.markdown(f"""
            <div class="card" style="height: 100%;">
                <div class="card-title">🎙 Audio Emotion Analysis</div>
                <div style="display:flex; justify-content:space-between; margin-bottom:15px; border-bottom:1px solid #2d333b; padding-bottom:15px;">
                    <div>
                        <div style="font-size:0.8rem; color:#8b949e; text-transform:uppercase;">Detected Emotion</div>
                        <div style="font-size:1.4rem; font-weight:600; color:{a_col}; margin-top:4px;">{a_emj} {audio.get("emotion", "N/A").upper()}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.8rem; color:#8b949e; text-transform:uppercase;">Confidence</div>
                        <div style="font-size:1.4rem; font-family:monospace; color:#c9d1d9; margin-top:4px;">{a_conf:.2f}</div>
                    </div>
                </div>
                <div style="font-size:0.9rem; color:#8b949e; margin-bottom:15px;">
                    Signal Quality: <span style="background:#222933; padding:2px 8px; border-radius:4px; color:#c9d1d9; font-family:monospace; margin-left:8px;">{audio.get("quality", 0.0):.2f}</span>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_video:
            v_conf = video.get("confidence", 0.0)
            v_col, v_emj = get_emotion_style(video.get("emotion", "N/A"))
            
            st.markdown(f"""
            <div class="card" style="height: 100%;">
                <div class="card-title">🎥 Video Emotion Analysis</div>
                <div style="display:flex; justify-content:space-between; margin-bottom:15px; border-bottom:1px solid #2d333b; padding-bottom:15px;">
                    <div>
                        <div style="font-size:0.8rem; color:#8b949e; text-transform:uppercase;">Detected Emotion</div>
                        <div style="font-size:1.4rem; font-weight:600; color:{v_col}; margin-top:4px;">{v_emj} {video.get("emotion", "N/A").upper()}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.8rem; color:#8b949e; text-transform:uppercase;">Confidence</div>
                        <div style="font-size:1.4rem; font-family:monospace; color:#c9d1d9; margin-top:4px;">{v_conf:.2f}</div>
                    </div>
                </div>
                <div style="font-size:0.9rem; color:#8b949e; margin-bottom:15px;">
                    Frame Quality: <span style="background:#222933; padding:2px 8px; border-radius:4px; color:#c9d1d9; font-family:monospace; margin-left:8px;">{video.get("quality", 0.0):.2f}</span>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # --- FUSION DECISION TRACE ---
        st.markdown("""
        <div class="card" style="margin-top: 24px;">
            <div class="card-title">🔬 Fusion Decision Flow</div>
        """, unsafe_allow_html=True)
        
        agreement_score = fusion.get("eri", {}).get("score", fused_conf)
        fusion_strategy = fusion.get("explanation", "Agreement Boost")
        if "High confidence video" in fusion_strategy: fusion_strategy = "Video Precedence"
        elif "High confidence audio" in fusion_strategy: fusion_strategy = "Audio Precedence"
        elif not fusion_strategy or fusion_strategy == "No explanation provided.": fusion_strategy = "Standard Weighted Fusion"
        
        a_str = f"{audio.get('emotion', 'N/A').upper()} ({a_conf:.2f})" if audio.get('emotion') != 'N/A' else "N/A"
        v_str = f"{video.get('emotion', 'N/A').upper()} ({v_conf:.2f})" if video.get('emotion') != 'N/A' else "N/A"
        
        st.markdown(f"""
        <div class="timeline">
           <div class="timeline-item"><span>🎙 Audio Prediction</span> <span>{a_str}</span></div>
           <div class="timeline-item"><span>🎥 Video Prediction</span> <span>{v_str}</span></div>
           <div class="timeline-item"><span>🤝 Agreement Score</span> <span>{agreement_score:.2f}</span></div>
           <div class="timeline-item"><span>⚙️ Rule Applied</span> <span style="font-family:inherit; color:#c9d1d9;">{fusion_strategy}</span></div>
           <div class="timeline-item final"><span>✅ Final Result</span> <span style="color:{color}; font-weight:bold; font-size:1.2rem;">{fused_emotion.upper()}</span></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ================= NEW UPGRADES SECTION =================
    st.markdown("---")
    
    col_u1, col_u2, col_u3 = st.columns(3, gap="large")
    
    with col_u1:
        # --- UPGRADE 1: EMOTION STABILITY MONITOR ---
        st.markdown("""
        <div class="card" style="height:100%;">
            <div class="card-title">⏳ Emotion Stability Monitor</div>
        """, unsafe_allow_html=True)
        
        temp_data = data.get("temporal_analysis", {})
        ess = temp_data.get("emotion_stability_score", 0.0)
        st_lvl = temp_data.get("stability_level", "UNKNOWN")
        dom_emo = temp_data.get("dominant_emotion", "N/A")
        
        st_color = "#00e6c3" if st_lvl == "HIGH" else "#f1c40f" if st_lvl == "MEDIUM" else "#e74c3c"
        
        st.markdown(f"""
        <div style="text-align:center; padding: 10px 0;">
            <div style="font-size:0.9rem; color:#8b949e; text-transform:uppercase; margin-bottom:5px;">Stability Level</div>
            <div style="font-size:1.8rem; font-weight:700; color:{st_color}; letter-spacing:1px;">{st_lvl}</div>
        </div>
        
        <div style="display:flex; justify-content:space-between; margin-top:15px; background:rgba(255,255,255,0.02); padding:15px; border-radius:8px;">
            <div>
                <p style="margin:0; font-size:0.8rem; color:#8b949e;">Dominant Target</p>
                <div style="font-size:1.1rem; font-weight:600; color:#c9d1d9;">{dom_emo.upper()}</div>
            </div>
            <div style="text-align:right;">
                <p style="margin:0; font-size:0.8rem; color:#8b949e;">ESS Score</p>
                <div style="font-size:1.1rem; font-family:monospace; color:{st_color};">{ess:.2f}</div>
            </div>
        </div>
        <div style="background:rgba(255,255,255,0.05); border-radius:4px; height:6px; margin-top:10px;">
            <div style="width:{ess*100}%; background:{st_color}; height:100%; border-radius:4px;"></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_u2:
        # --- UPGRADE 3: FUSION STRATEGY INTELLIGENCE ---
        st.markdown("""
        <div class="card" style="height:100%;">
            <div class="card-title">♟️ Fusion Strategy Intelligence</div>
        """, unsafe_allow_html=True)
        
        dyn_strategy = fusion.get("fusion_strategy", "weighted_fusion")
        
        strat_desc = {
            "agreement_boost": ("Agreement Boost", "Confirmed cross-modal consensus > 0.75"),
            "audio_dominance": ("Audio Dominance", "Audio confidence drastically outweights Video"),
            "audio_priority": ("Audio Priority", "Video quality severely compromised (<0.40)"),
            "weighted_fusion": ("Weighted Fusion", "Standard soft multimodal summation")
        }
        
        s_title, s_desc = strat_desc.get(dyn_strategy, ("Standard Fusion", "Fallback applied"))
        
        st.markdown(f"""
        <div style="text-align:center; padding: 15px 0;">
            <div style="display:inline-block; padding:8px 16px; border-radius:20px; border:1px solid #00e6c3; background:rgba(0,230,195,0.1); color:#00e6c3; font-weight:600; font-size:1.1rem; margin-bottom:15px;">
                {s_title}
            </div>
            <p style="color:#c9d1d9; font-size:0.95rem; line-height:1.5;">{s_desc}</p>
        </div>
        <div style="margin-top:auto; font-family:monospace; color:#8b949e; text-align:center; font-size:0.85rem; background:#222933; padding:8px; border-radius:6px;">
            sys.internal.strategy_id: {dyn_strategy.upper()}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_u3:
        # --- UPGRADE 2: EMOTION DISTRIBUTION ANALYSIS ---
        st.markdown("""
        <div class="card" style="height:100%;">
            <div class="card-title">📊 Emotion Distribution Analysis</div>
        """, unsafe_allow_html=True)
        
        ce = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD', 'SUR']
        colors = ["#e74c3c", "#94a3b8", "#9b59b6", "#f1c40f", "#95a5a6", "#3498db", "#e67e22"]
        
        a_probs = audio.get("probabilities", [])
        v_probs = video.get("probabilities", [])
        
        if len(a_probs) == 7 and len(v_probs) == 7:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=ce, x=a_probs, name='Audio', orientation='h', 
                marker_color='rgba(0, 230, 195, 0.7)'
            ))
            fig.add_trace(go.Bar(
                y=ce, x=v_probs, name='Video', orientation='h',
                marker_color='rgba(52, 152, 219, 0.7)'
            ))
            
            fig.update_layout(
                barmode='group',
                height=220,
                margin=dict(l=0, r=0, t=20, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8b949e", size=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("<p style='color:#8b949e; text-align:center;'>Distributions unavailable for current pipeline.</p>", unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

    # --- DEVELOPER DEBUG PANEL ---
    st.markdown("<h3 style='color:#8b949e; font-size:1.1rem; margin-top:20px;'>Developer Debug Panel</h3>", unsafe_allow_html=True)
    with st.expander("Show Model Output"):
        st.json(data)

def create_radial(conf, title, color):
    fig = go.Figure(go.Pie(
        values=[conf, max(0, 1-conf)],
        hole=0.8,
        marker_colors=[color, "rgba(255,255,255,0.05)"],
        textinfo="none",
        hoverinfo="none"
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(t=0,b=0,l=0,r=0),
        height=180,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text=f"{conf:.2f}", showarrow=False, font=dict(size=28, color="white", family="Inter"))]
    )
    return fig

def create_eri_gauge(score, level):
    color = "#f1c40f" if level.upper() in ["MEDIUM", "MODERATE"] else "#e74c3c"
    if level.upper() == "HIGH": color = "#00e6c3"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 1], 'tickwidth': 0, 'showticklabels': False},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
        },
        number = {'font': {'size': 36, 'color': "white"}}
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=160,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    return fig, color

# ------------------ TOP HEADER ------------------
st.markdown("""
<div class="header-container">
    <div>
        <h1 style="margin-bottom:0; font-size:2.5rem; font-weight:700;">AMCEF — Adaptive Multi-modal Emotion Intelligence</h1>
        <h3 style="margin-top:5px; color:#8b949e; font-weight:400;">Audio–Visual Emotion Fusion System</h3>
    </div>
    <div style="text-align:right;">
        <div class="status-indicator" style="margin-bottom:8px; color:#8b949e;">
            <span class="status-dot" style="background-color:#c9d1d9; box-shadow:none;"></span> API Connected
        </div>
        <br>
        <div class="status-indicator" style="color:#00e6c3; font-weight:700; text-shadow: 0 0 10px rgba(0,230,195,0.4); border:1px solid #00e6c3; padding:4px 12px; border-radius:15px; background:rgba(0,230,195,0.05);">
            <span class="status-dot" style="background-color:#00e6c3; box-shadow: 0 0 8px #00e6c3;"></span> Model Loaded
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

backend_url = "http://localhost:8000"

# ------------------ LEFT PANEL (SIDEBAR) ------------------
with st.sidebar:
    st.markdown("<h2 style='color:white;'>🎛️ Emotion Input Center</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    mode = st.radio(
        "Mode Selector",
        ["Upload Audio & Video", "Live Audio", "Live Camera"],
    )
    
    audio_file = None
    video_file = None
    
    if mode == "Upload Audio & Video":
        st.markdown("#### Audio Upload", help="Accepts WAV format")
        audio_file = st.file_uploader("Accepts WAV", type=["wav"], label_visibility="collapsed")
        
        st.markdown("#### Video Upload", help="Accepts MP4 / FLV / MPEG4 format")
        video_file = st.file_uploader("Accepts MP4 / FLV / MPEG4", type=["mp4", "flv", "mpeg4", "avi"], label_visibility="collapsed")
        
        if audio_file or video_file:
            st.markdown("""
            <div style="background:#222933; padding:15px; border-radius:8px; margin-top:20px; font-size:0.9rem; color:#c9d1d9; border: 1px solid #2d333b;">
                <div style="font-weight:600; margin-bottom:10px; color:#8b949e; letter-spacing:1px; font-size:0.8rem;">FILE METADATA</div>
            """, unsafe_allow_html=True)
            if audio_file:
                st.markdown(f"**Filename:** `{audio_file.name}`<br>**Size:** {audio_file.size / 1024:.1f} KB<br>**Duration:** ~ 00:04", unsafe_allow_html=True)
                st.markdown("---")
            if video_file:
                st.markdown(f"**Filename:** `{video_file.name}`<br>**Size:** {video_file.size / (1024*1024):.2f} MB<br>**Duration:** ~ 00:10", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    elif mode == "Live Audio":
        st.info("💡 **Browser-Based Recording**: Capturing audio directly via browser. No system-level `ffmpeg` required.")
        recorded_audio = mic_recorder(
            start_prompt="🎙️ Start Recording",
            stop_prompt="🛑 Stop Recording",
            key='recorder'
        )
        if recorded_audio:
            # mic_recorder returns a dict with 'bytes', 'sample_rate', 'sample_width'
            # We create a named object similar to st.audio_input for compatibility
            from io import BytesIO
            audio_file = BytesIO(recorded_audio['bytes'])
            audio_file.name = "live_recording.wav"
        
    elif mode == "Live Camera":
        video_file = st.camera_input("Capture Image")

    st.markdown("<br>", unsafe_allow_html=True)
    run_clicked = st.button("🚀 Run Emotion Analysis", use_container_width=True, type="primary")

# ------------------ MAIN CONTENT ------------------
if run_clicked and (audio_file or video_file):
    with st.spinner("Analyzing cross-modal features..."):
        files = {}
        if audio_file:
            files["audio_file"] = (audio_file.name, audio_file.getvalue(), "audio/wav")
        if video_file:
            files["video_file"] = (video_file.name, video_file.getvalue(), "video/mp4")
            
        try:
            response = requests.post(f"{backend_url}/predict", files=files)
            if response.status_code == 200:
                data = response.json()
                
                # Check for face detection warning
                if data.get("video") and not data["video"].get("face_detected", True):
                    st.warning("⚠️ **No Face Detected**: Please ensure your face is clearly visible to the camera for accurate analysis.")
                    st.stop()
                
            else:
                st.error(f"Backend Error: {response.text}")
                st.stop()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend API at http://localhost:8000")
            st.stop()

    fusion = data.get("fusion", {})
    audio = data.get("audio")
    video = data.get("video")

    fused_emotion = fusion.get("final_emotion", "N/A")
    fused_conf = fusion.get("final_confidence", 0.0)
    color, emoji = get_emotion_style(fused_emotion)
    
    # Defaults if missing
    if not audio:
        audio = {"emotion": "N/A", "confidence": 0.0, "quality": 0.0}
    if not video:
        video = {"emotion": "N/A", "confidence": 0.0, "quality": 0.0}

    col_c, col_r = st.columns([1, 1.4], gap="large")
    
    with col_c:
        # --- CENTER PANEL: Emotion Prediction Card ---
        st.markdown(f"""
        <div class="card" style="border-top: 4px solid {color}; box-shadow: 0 -4px 12px {color}22;">
            <div class="card-title">🎯 Emotion Prediction Card</div>
            <div style="text-align:center; padding: 15px 0;">
                <h3 style="color:#8b949e; font-weight:500; font-size:1rem; margin-bottom:5px;">Detected Emotion</h3>
                <h1 style="font-size:3.5rem; color:{color}; margin:0; text-shadow: 0 0 16px {color}44;">{emoji} {fused_emotion.upper()}</h1>
            </div>
            <div style="margin-top:10px; margin-bottom:20px;">
                <div style="display:flex; justify-content:space-between; color:#c9d1d9; font-size:0.9rem; margin-bottom:5px;">
                    <span>Confidence Meter</span>
                    <span style="color:{color}; font-weight:bold;">{fused_conf:.2f}</span>
                </div>
                <div style="background:rgba(255,255,255,0.05); border-radius:4px; height:8px; overflow:hidden;">
                    <div style="width:{fused_conf*100}%; background:{color}; height:100%; border-radius:4px; transition:width 1s; box-shadow: 0 0 8px {color};"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        # Radial Gauge for Fusion Confidence inside the same visually connected layout
        st.plotly_chart(create_radial(fused_conf, "Fusion Confidence", color), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # --- FUSION INTELLIGENCE SECTION ---
        st.markdown("""
        <div class="card">
            <div class="card-title">🧠 Fusion Intelligence</div>
        """, unsafe_allow_html=True) 
        eri = fusion.get("eri", {})
        eri_score = eri.get("score", fused_conf)
        eri_level = eri.get("level", "LOW")
        
        fig_eri, eri_col = create_eri_gauge(eri_score, eri_level)
        st.markdown(f"<p style='text-align:center; color:#8b949e; margin-bottom:0;'>Emotion Reliability Index (ERI)</p>", unsafe_allow_html=True)
        st.plotly_chart(fig_eri, use_container_width=True)
        st.markdown(f"<div style='text-align:center; font-size:1.5rem; font-weight:700; color:{eri_col}; text-shadow: 0 0 10px {eri_col}55;'>{eri_level.upper()}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # --- CONSISTENCY ANALYSIS ---
        st.markdown("""
        <div class="card">
            <div class="card-title">⚖️ Emotion Consistency Monitor</div>
        """, unsafe_allow_html=True)
        
        a_emo = audio.get("emotion", "").lower()
        v_emo = video.get("emotion", "").lower()
        
        conflict = False
        if a_emo != "n/a" and v_emo != "n/a":
            if a_emo != v_emo: conflict = True
            
        status_text = "⚠ Emotional Conflict Detected" if conflict else "✅ Modalities Agree"
        status_color = "#e74c3c" if conflict else "#00e6c3"
        agreement_score = eri_score # ERI acts as the agreement metric
        
        st.markdown(f"""
        <div style="margin: 15px 0;">
            <p style="color:#8b949e; margin-bottom:5px;">Agreement Score</p>
            <h2 style="margin:0; font-family:monospace; color:#c9d1d9;">{agreement_score:.2f}</h2>
        </div>
        <div style="padding:14px 18px; border-radius:8px; background:rgba({ '231,76,60' if conflict else '0,230,195' }, 0.1); border:1px solid {status_color}; text-align:center;">
            <span style="color:{status_color}; font-weight:600; font-size:1.1rem;">Status: {status_text}</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        # --- RIGHT PANEL ---
        col_audio, col_video = st.columns(2)
        
        with col_audio:
            a_conf = audio.get("confidence", 0.0)
            a_col, a_emj = get_emotion_style(audio.get("emotion", "N/A"))
            
            st.markdown(f"""
            <div class="card" style="height: 100%;">
                <div class="card-title">🎙 Audio Emotion Analysis</div>
                <div style="display:flex; justify-content:space-between; margin-bottom:15px; border-bottom:1px solid #2d333b; padding-bottom:15px;">
                    <div>
                        <div style="font-size:0.8rem; color:#8b949e; text-transform:uppercase;">Detected Emotion</div>
                        <div style="font-size:1.4rem; font-weight:600; color:{a_col}; margin-top:4px;">{a_emj} {audio.get("emotion", "N/A").upper()}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.8rem; color:#8b949e; text-transform:uppercase;">Confidence</div>
                        <div style="font-size:1.4rem; font-family:monospace; color:#c9d1d9; margin-top:4px;">{a_conf:.2f}</div>
                    </div>
                </div>
                <div style="font-size:0.9rem; color:#8b949e; margin-bottom:15px;">
                    Signal Quality: <span style="background:#222933; padding:2px 8px; border-radius:4px; color:#c9d1d9; font-family:monospace; margin-left:8px;">{audio.get("quality", 0.0):.2f}</span>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_video:
            v_conf = video.get("confidence", 0.0)
            v_col, v_emj = get_emotion_style(video.get("emotion", "N/A"))
            
            st.markdown(f"""
            <div class="card" style="height: 100%;">
                <div class="card-title">🎥 Video Emotion Analysis</div>
                <div style="display:flex; justify-content:space-between; margin-bottom:15px; border-bottom:1px solid #2d333b; padding-bottom:15px;">
                    <div>
                        <div style="font-size:0.8rem; color:#8b949e; text-transform:uppercase;">Detected Emotion</div>
                        <div style="font-size:1.4rem; font-weight:600; color:{v_col}; margin-top:4px;">{v_emj} {video.get("emotion", "N/A").upper()}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.8rem; color:#8b949e; text-transform:uppercase;">Confidence</div>
                        <div style="font-size:1.4rem; font-family:monospace; color:#c9d1d9; margin-top:4px;">{v_conf:.2f}</div>
                    </div>
                </div>
                <div style="font-size:0.9rem; color:#8b949e; margin-bottom:15px;">
                    Frame Quality: <span style="background:#222933; padding:2px 8px; border-radius:4px; color:#c9d1d9; font-family:monospace; margin-left:8px;">{video.get("quality", 0.0):.2f}</span>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # --- FUSION DECISION TRACE ---
        st.markdown("""
        <div class="card" style="margin-top: 24px;">
            <div class="card-title">🔬 Fusion Decision Flow</div>
        """, unsafe_allow_html=True)
        
        fusion_strategy = fusion.get("explanation", "Agreement Boost")
        if "High confidence video" in fusion_strategy: fusion_strategy = "Video Precedence"
        elif "High confidence audio" in fusion_strategy: fusion_strategy = "Audio Precedence"
        elif not fusion_strategy or fusion_strategy == "No explanation provided.": fusion_strategy = "Standard Weighted Fusion"
        
        a_str = f"{audio.get('emotion', 'N/A').upper()} ({a_conf:.2f})" if audio.get('emotion') != 'N/A' else "N/A"
        v_str = f"{video.get('emotion', 'N/A').upper()} ({v_conf:.2f})" if video.get('emotion') != 'N/A' else "N/A"
        
        st.markdown(f"""
        <div class="timeline">
           <div class="timeline-item"><span>🎙 Audio Prediction</span> <span>{a_str}</span></div>
           <div class="timeline-item"><span>🎥 Video Prediction</span> <span>{v_str}</span></div>
           <div class="timeline-item"><span>🤝 Agreement Score</span> <span>{agreement_score:.2f}</span></div>
           <div class="timeline-item"><span>⚙️ Fusion Strategy</span> <span style="font-family:inherit; color:#c9d1d9;">{fusion_strategy}</span></div>
           <div class="timeline-item final"><span>✅ Final Result</span> <span style="color:{color}; font-weight:bold; font-size:1.2rem;">{fused_emotion.upper()}</span></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- DEVELOPER DEBUG PANEL ---
    st.markdown("<h3 style='color:#8b949e; font-size:1.1rem; margin-top:20px;'>Developer Debug Panel</h3>", unsafe_allow_html=True)
    with st.expander("Show Model Output"):
        st.json(data)
        
else:
    # Empty placeholder when no analysis is run yet
    st.markdown("""
    <div style="display:flex; height:60vh; justify-content:center; align-items:center; flex-direction:column; color:#8b949e;">
        <div style="font-size:5rem; margin-bottom:20px; text-shadow: 0 0 20px rgba(0,230,195,0.2);">📥</div>
        <h2 style="font-weight:400; color:#c9d1d9;">Awaiting AI Analysis</h2>
        <p style="color:#8b949e;">Please upload audio and video files from the Emotion Input Center to begin the fusion process.</p>
    </div>
    """, unsafe_allow_html=True)
