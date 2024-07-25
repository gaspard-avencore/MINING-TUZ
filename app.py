import streamlit as st
import ast
import math
import threading
import http.server
import socketserver
import webbrowser
import utils.parameters as parameters
import utils.processing as processing

import subprocess

PORT = 8001
DIRECTORY = "custom_app"

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def start_server():
    handler = CustomHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        httpd.serve_forever()

# def open_custom_app():
#     url = f"http://localhost:{PORT}"
#     webbrowser.open_new_tab(url)


def start_flask_server():
    subprocess.Popen(['python', 'flask_server.py'])

def open_custom_app():
    start_flask_server()

def launch_simulation(hexagons_to_draw, hexagonal_structure, hex_size, vertices, tuz_capa=None):
    structure = processing.plot_hexagonal_structure(hex_size, hexagons_to_draw)
    st.session_state.structure = structure
    
    if type(tuz_capa)==int:
        N_tuz = math.ceil(len(hexagons_to_draw) / tuz_capa)
        hexagonal_structure, tuzs = processing.get_positions_tuz(hexagonal_structure, N_tuz)
        injection_lines, extraction_lines = processing.get_lines(hexagonal_structure, vertices, tuzs)
        costs = processing.get_costs(tuzs, injection_lines, extraction_lines, parameters.CAPEX)
        value = processing.get_value(hexagonal_structure, parameters.VALUES)
        structure_with_tuzs = processing.plot_cells_tuzs_lines(hexagonal_structure, tuzs, injection_lines, extraction_lines)
        summary_str = processing.print_summary(hexagons_to_draw, tuzs, costs, injection_lines, extraction_lines, value)
        tuz_sizes = processing.plot_tuz_sizes(tuzs, standards=[10, 20])
        st.session_state.structure_with_tuzs = structure_with_tuzs
        st.session_state.summary_str = summary_str
        st.session_state.tuz_sizes = tuz_sizes
        
    else:
        elements = processing.tune_tuz_capa(
            hexagons_to_draw, 
            hexagonal_structure, 
            vertices, 
            parameters.CAPEX, 
            parameters.VALUES, 
            min_capa=10, 
            maxcapa=60 
        )
        stacked_costs_fig, avg_distances_fig = processing.results_tuning(hexagons_to_draw, elements)
        st.session_state.stacked_costs_fig = stacked_costs_fig
        st.session_state.avg_distances_fig = avg_distances_fig

# Configuration
st.set_page_config(
    page_title="Mining TUZ",
    page_icon="🪱",
    layout='wide'
)
hide_decoration_bar_style = '''
    <style>
        header {visibility: hidden;}
    </style>
'''
hide_streamlit_style = '''
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
'''
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)


st.title("TUZ Modeling Tool")

if st.button("Configure mine geography"):
    threading.Thread(target=start_server, daemon=True).start()
    open_custom_app()
    st.success(f"Custom app is being served at http://localhost:{PORT}")

mine_geography = st.file_uploader("Upload mine geography", type="txt")
if mine_geography is not None:
    content = mine_geography.read().decode("utf-8")
    hexagons_to_draw = ast.literal_eval("["+content+"]")
    hexagons_to_draw = [hex_coords + (0, ) for hex_coords in hexagons_to_draw]

    hex_size = st.number_input("Cells size", value=40, min_value=1, step=1)

    if hex_size:
        st.session_state.hex_size = hex_size
        hexagonal_structure = processing.get_hexagonal_structure(hex_size, hexagons_to_draw)
        st.session_state.hexagonal_structure = hexagonal_structure
        progress = 0.
        progress_bar = st.progress(progress)
        for result in processing.get_vertices(hexagonal_structure):
            if type(result)==float:
                progress = result
                progress_bar.progress(progress)
            else : 
                progress_bar.progress(1.)
                vertices = result
                st.session_state.vertices = vertices
    
    
    # launch and compute results
    
    if "hexagonal_structure" in st.session_state and "vertices" in st.session_state:
        hexagonal_structure = st.session_state["hexagonal_structure"]
        vertices = st.session_state["vertices"]
        
        col1, col2 = st.columns(2)
        target_tuz_capa = col1.number_input("Tuz Capacity", value=17, min_value=1, step=1)
        launch_btn_single = col1.button("Launch simulation", use_container_width=True, type="primary")
        
        col2_1, col2_2 = col2.columns(2)
        col2_1.number_input("Min Capa", value=10, min_value=1, step=1)
        col2_2.number_input("Max Capa", value=60, min_value=1, step=1)
        launch_btn_tuning = col2.button("Launch tuning", use_container_width=True, type="primary")
        
        if target_tuz_capa and launch_btn_single:
            launch_simulation(hexagons_to_draw, hexagonal_structure, hex_size, vertices, tuz_capa=target_tuz_capa)
            st.session_state.show_result = "single"
        elif launch_btn_tuning:
            launch_simulation(hexagons_to_draw, hexagonal_structure, hex_size, vertices)
            st.session_state.show_result = "tuning"
    
    
    # display results
    
    if "show_result" in st.session_state :
        if st.session_state.show_result=="single" and all([x in st.session_state for x in ["structure_with_tuzs", "structure", "summary_str", "tuz_sizes"]]):
            with st.expander("Map"):
                display_tuzs = st.toggle("Display TUZs")
                if display_tuzs:
                    st.pyplot(st.session_state.structure_with_tuzs, use_container_width=False)
                else: 
                    st.pyplot(st.session_state.structure, use_container_width=False)
            with st.expander("TUZ sizes"):
                st.pyplot(st.session_state.tuz_sizes, use_container_width=True)
            with st.expander("Summary"):
                st.write(st.session_state.summary_str)
        
        if st.session_state.show_result=="tuning" and "stacked_costs_fig" in st.session_state and "avg_distances_fig" in st.session_state :
            with st.expander("Comparison"):
                st.pyplot(st.session_state.stacked_costs_fig)
                st.pyplot(st.session_state.avg_distances_fig)
    
