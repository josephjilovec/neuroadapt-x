import streamlit as st
import time
import numpy as np
import threading
from typing import Dict, Any

# --- MOCK COMMUNICATION INTERFACE ---
# In a full deployment, the RealTimeProcessor would write to a multiprocessing.Queue,
# LSL outlet, or shared memory that this thread monitors.
# We use a global dictionary protected by a lock to simulate shared state.

SHARED_STATE: Dict[str, Any] = {
    'command': 'NEUTRAL',
    'confidence': 0.5,
    'adaptation_active': False,
    'adaptation_buffer_size': 0,
    'processor_running': False
}
STATE_LOCK = threading.Lock()

def mock_realtime_processor_loop():
    """
    Mocks the RealTimeProcessor thread/process that updates the BCI state.
    
    This loop simulates a 60-second test run:
    1. Baseline (0-10s): High confidence, Neutral/LEFT commands.
    2. Stress Period (10-30s): Low confidence, Activation of Adaptation.
    3. Recovery (30-40s): Confidence slowly rises, Adaptation buffer clears.
    4. Post-Adaptation (40s+): High confidence, accurate commands resume.
    """
    global SHARED_STATE
    
    def update_state(cmd, conf, adapt, buf_size, running=True):
        with STATE_LOCK:
            SHARED_STATE.update({
                'command': cmd,
                'confidence': conf,
                'adaptation_active': adapt,
                'adaptation_buffer_size': buf_size,
                'processor_running': running
            })

    start_time = time.time()
    st.session_state.processor_running = True
    update_state('STARTING', 0.5, False, 0)
    print("Mock Processor Started.")

    try:
        while st.session_state.processor_running:
            elapsed = time.time() - start_time

            if elapsed < 10:
                # 1. Baseline: Good performance, no adaptation
                cmd = 'LEFT' if int(elapsed * 2) % 2 == 0 else 'RIGHT'
                update_state(cmd, np.random.uniform(0.85, 0.95), False, 0)
            
            elif elapsed < 30:
                # 2. Stress Period: Confidence drops, Adaptation required
                stress_conf = np.random.uniform(0.5, 0.65)
                buf = min(20, int(elapsed - 10) * 2)
                update_state('NEUTRAL', stress_conf, True, buf)
                if elapsed > 25: # Mock adaptation step
                    update_state('NEUTRAL', 0.70, True, 0) # Adaptation step clears buffer

            elif elapsed < 40:
                # 3. Recovery: Confidence climbs back up
                recovery_conf = np.clip(0.70 + (elapsed - 30) * 0.05, 0.7, 0.9)
                update_state('RIGHT', recovery_conf, False, 0)
            
            else:
                # 4. Post-Adaptation: Resumed high performance
                cmd = 'LEFT' if int(elapsed) % 2 == 0 else 'RIGHT'
                update_state(cmd, np.random.uniform(0.9, 0.98), False, 0)

            time.sleep(0.1)

    except Exception as e:
        print(f"Mock processor error: {e}")
    finally:
        update_state('STOPPED', 0.0, False, 0, running=False)
        print("Mock Processor Stopped.")

# --- STREAMLIT UI ---

st.set_page_config(
    page_title="NeuroAdapt-X Live Demo",
    layout="wide"
)

def get_current_state() -> Dict[str, Any]:
    """Safely reads the shared state."""
    with STATE_LOCK:
        return SHARED_STATE.copy()

def start_processor():
    """Starts the mock BCI processor thread."""
    if st.session_state.get('processor_thread') is None or not st.session_state.processor_thread.is_alive():
        st.session_state.processor_running = True
        thread = threading.Thread(target=mock_realtime_processor_loop, daemon=True)
        thread.start()
        st.session_state.processor_thread = thread
        st.session_state.last_update_time = time.time()
    else:
        st.warning("Processor is already running.")

def stop_processor():
    """Stops the mock BCI processor thread."""
    st.session_state.processor_running = False
    
def main_app():
    """Renders the main Streamlit application interface."""
    
    st.title("🛰️ NeuroAdapt-X: Stress-Resilient Neural Decoder")
    st.markdown("Real-time classification and adaptive model visualization for simulated space operations.")

    if 'processor_running' not in st.session_state:
        st.session_state.processor_running = False
    if 'processor_thread' not in st.session_state:
        st.session_state.processor_thread = None
    if 'command_history' not in st.session_state:
        st.session_state.command_history = []
    
    # --- 1. Control Panel ---
    col_start, col_stop, col_status = st.columns([1, 1, 3])
    
    if col_start.button("🚀 Start BCI Processor", disabled=st.session_state.processor_running):
        start_processor()
        st.experimental_rerun()
        
    if col_stop.button("🛑 Stop Processor", disabled=not st.session_state.processor_running):
        stop_processor()
        st.session_state.command_history.clear()
        st.experimental_rerun()

    status_color = "green" if st.session_state.processor_running else "red"
    col_status.markdown(
        f"**Processor Status:** :{'green_circle:' if st.session_state.processor_running else 'red_circle:'} "
        f"{'RUNNING' if st.session_state.processor_running else 'STOPPED'}",
        unsafe_allow_html=True
    )

    # --- 2. Real-Time Metrics ---
    st.subheader("Real-Time Decoding & Adaptation Metrics")
    state = get_current_state()
    
    cmd_text = state['command']
    confidence_value = state['confidence']
    adaptation_status = state['adaptation_active']
    buffer_size = state['adaptation_buffer_size']
    
    # Update command history
    if st.session_state.processor_running and cmd_text != 'NEUTRAL' and (time.time() - st.session_state.get('last_cmd_log', 0) > 0.5):
        st.session_state.command_history.insert(0, (cmd_text, confidence_value))
        st.session_state.command_history = st.session_state.command_history[:10] # Keep last 10
        st.session_state.last_cmd_log = time.time()
    
    col_cmd, col_conf, col_adapt, col_buffer = st.columns(4)

    # Command Visualization
    cmd_style = 'font-size: 24px; font-weight: bold;'
    if cmd_text == 'LEFT':
        cmd_icon = "⬅️"
    elif cmd_text == 'RIGHT':
        cmd_icon = "➡️"
    else:
        cmd_icon = "⚫"
        cmd_text = "NEUTRAL"
        
    col_cmd.metric(
        label="BCI Command",
        value=f"{cmd_icon} {cmd_text}",
    )
    
    # Confidence Gauge
    conf_delta = f"{confidence_value * 100:.1f}%"
    conf_color = "inverse" if confidence_value < 0.75 else "normal"
    col_conf.metric("Model Confidence", value=f"{confidence_value:.2f}", delta_color=conf_color)
    
    # Adaptation Status
    adapt_icon = "⚡" if adaptation_status else "💤"
    adapt_color = "red" if adaptation_status else "green"
    col_adapt.markdown(f"**Adaptation Status**", help="Adaptation (CORAL Loss) engages when confidence drops.")
    col_adapt.markdown(
        f"<p style='color:{adapt_color}; font-weight: bold;'>{adapt_icon} {'ADAPTING' if adaptation_status else 'STABLE'}</p>",
        unsafe_allow_html=True
    )
    
    # Adaptation Buffer
    col_buffer.metric(
        label="Adaptation Buffer Size",
        value=buffer_size,
        help="Number of low-confidence epochs collected for unsupervised adaptation."
    )

    # --- 3. Command History & Rover Simulation (Simplified) ---
    st.subheader("Command Log")
    
    if st.session_state.command_history:
        log_text = "\n".join([f"- {cmd} (Conf: {conf:.2f})" for cmd, conf in st.session_state.command_history])
        st.code(log_text)
    else:
        st.info("Start the processor to see the decoded BCI commands log.")
        
    st.markdown("""---""")
    st.markdown("""
    ### Rover Status Simulation (Conceptual)
    In the full demo, the `rover_control.py` Pygame window would be updated by the BCI command. 
    Here, the movement is visualized by the command output above.
    """)

    # --- Auto-refresh mechanism (Crucial for live data) ---
    if st.session_state.processor_running:
        # Rerun the script every 100ms to fetch latest state
        time.sleep(0.1)
        st.experimental_rerun()

if __name__ == '__main__':
    # Initialize processor state for the first run
    if 'processor_running' not in st.session_state:
        st.session_state.processor_running = False
        
    # Check if the thread died unexpectedly and clean up session state
    if st.session_state.get('processor_thread') and not st.session_state.processor_thread.is_alive():
        st.session_state.processor_running = False
        st.session_state.processor_thread = None
        
    main_app()
    
# NOTE: To run this file, you need Streamlit installed: `pip install streamlit`.
# Run the demo using: `streamlit run src/demo/web_demo.py`
