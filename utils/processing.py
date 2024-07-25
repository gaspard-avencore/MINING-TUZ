import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
import math
from tqdm import tqdm


# map

def get_hexagon(center, size):
    """Generate the vertices of a hexagon given a center and size."""
    angles = np.linspace(0, 2 * np.pi, 7)
    x_hex = center[0] + size * np.cos(angles)
    y_hex = center[1] + size * np.sin(angles)
    z_hex = center[2] + np.zeros(7)
    return x_hex, y_hex, z_hex

def get_hexagonal_structure(size, hexagons):
    hexagonal_structure = {}
    h_dist = 3/2 * size
    v_dist = np.sqrt(3) * size
    id_hex = 0
    for col, row, h in hexagons:
        x = col * h_dist
        y = row * v_dist
        if col % 2 == 1:
            y += v_dist / 2
        x_hex, y_hex, z_hex = get_hexagon((x, y, h), size)
        hexagonal_structure.update({
            id_hex: {
                "x_hex": x_hex, "y_hex": y_hex, "z_hex": z_hex, 
                "x_center": np.mean(x_hex), "y_center": np.mean(y_hex), "z_center": np.mean(z_hex)
            }
        })
        id_hex += 1
    
    return hexagonal_structure

def plot_hexagonal_structure(size, hexagons_to_draw):
    """Plot a hexagonal structure drawing only specified hexagons."""
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    hex_structure = get_hexagonal_structure(size, hexagons_to_draw)
    for hex in hex_structure.values():
        x_hex, y_hex = hex["x_hex"], hex["y_hex"]
        ax.plot(x_hex, y_hex, '-', color='black')
    
    plt.axis('off')
    return fig


# Tuz positions

def get_positions_tuz(hexagonal_structure, N_tuz):
    X = [[hexagon["x_center"], hexagon["y_center"]] for hexagon in hexagonal_structure.values()]
    kmeans =  KMeans(N_tuz, random_state=0, n_init="auto").fit(X)
    labels =  kmeans.labels_
    centroids = kmeans.cluster_centers_
    for i, hexagon in hexagonal_structure.items():
        hexagon.update({
            "cluster_id": labels[i], 
            "x_cluster_center": centroids[labels[i]][0], 
            "y_cluster_center": centroids[labels[i]][1]
        })
    tuzs = {
        i: {"x": centroid[0], 
            "y": centroid[1], 
            "z": 0, # to be updated according to data altitude
            "injection_points": [], 
            "extraction_points": [], 
        } for i, centroid in enumerate(centroids)} 
    
    return hexagonal_structure, tuzs

def plot_cells_tuzs(hexagonal_structure, tuzs):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    N_tuz = len(tuzs)
    cmap = ListedColormap(plt.get_cmap('viridis')(np.linspace(0, 1, N_tuz)))
    colors = [cmap(i) for i in range(N_tuz)]
    
    for hex in hexagonal_structure.values():
        x_hex, y_hex = hex["x_hex"], hex["y_hex"]
        ax.plot(x_hex, y_hex, '-', color=colors[hex["cluster_id"]])
    
    for i, tuz in tuzs.items(): 
        ax.plot(tuz["x"], tuz["y"], '+', color=colors[i])
    
    plt.axis('off')
    return fig


# costs modeling

def get_distance(x1, y1, z1, x2, y2, z2):
    return np.linalg.norm(np.array([x1, y1, z1]) - np.array([x2, y2, z2]))

def get_vertices(hexagonal_structure, tolerance=1e-2):
    vertices = {}
    vertex_id = 0
    vertex_map = {}
    iterator = 0
    
    for hex_id, hex_data in hexagonal_structure.items():
        yield iterator / len(hexagonal_structure)
        
        x_hex = hex_data["x_hex"]
        y_hex = hex_data["y_hex"]
        z_hex = hex_data["z_hex"]

        for i in range(len(x_hex)):
            vertex = (x_hex[i], y_hex[i], z_hex[i])
            matched = False

            for existing_vertex in vertex_map:
                if np.allclose(vertex, existing_vertex, atol=tolerance):
                    vertices[vertex_map[existing_vertex]]["related_hexagons_ids"].append(hex_id)
                    matched = True
                    break

            if not matched:
                vertex_map[vertex] = vertex_id
                vertices[vertex_id] = {
                    "vertex_position_x": x_hex[i],
                    "vertex_position_y": y_hex[i],
                    "vertex_position_z": z_hex[i],
                    "related_hexagons_ids": [hex_id]
                }
                vertex_id += 1
        iterator+=1
    for vertex in vertices.values():
        vertex["related_hexagons_ids"] = list(np.unique(vertex["related_hexagons_ids"]))
    yield vertices

def get_lines(hexagonal_structure, vertices, tuzs):
    extraction_lines = {
        hexagon_id: {
            "x_from": tuzs[hexagon_info["cluster_id"]]["x"],
            "y_from": tuzs[hexagon_info["cluster_id"]]["y"],
            "z_from": tuzs[hexagon_info["cluster_id"]]["z"],
            
            "x_to": hexagon_info["x_center"], 
            "y_to": hexagon_info["y_center"], 
            "z_to": hexagon_info["z_center"], 
        }
        for hexagon_id, hexagon_info in hexagonal_structure.items()
    }
    for hexagon_id, hexagon_info in hexagonal_structure.items():
        tuzs[hexagon_info["cluster_id"]]["extraction_points"].append(hexagon_id)
    
    injection_lines = {}
    for i, vertex in vertices.items():
        hexagon_ids = vertex["related_hexagons_ids"]
        candidate_tuzs_ids = list(set(hexagonal_structure[hexagon_id]["cluster_id"] for hexagon_id in hexagon_ids))
        candidate_tuzs = {key: val for key, val in tuzs.items() if key in candidate_tuzs_ids}
        distances_tuzs = {
            tuz_id: get_distance(
                vertex["vertex_position_x"], vertex["vertex_position_y"], vertex["vertex_position_z"], 
                tuz_info["x"], tuz_info["y"], tuz_info["z"]
            )
        for tuz_id, tuz_info in candidate_tuzs.items()
        }
        selected_tuz_id = next(iter(distances_tuzs)) if len(distances_tuzs)==1 else min(distances_tuzs, key=distances_tuzs.get)
        tuzs[selected_tuz_id]["injection_points"].append(i)
        injection_lines.update({
            i : {
                "id_from": i, 
                
                "x_from": vertex["vertex_position_x"], 
                "y_from": vertex["vertex_position_y"], 
                "z_from": vertex["vertex_position_z"], 
                
                "x_to": tuzs[selected_tuz_id]["x"], 
                "y_to": tuzs[selected_tuz_id]["y"], 
                "z_to": tuzs[selected_tuz_id]["z"], 
            },
        })
    return injection_lines, extraction_lines

def plot_cells_tuzs_lines(hexagonal_structure, tuzs, injection_lines, extraction_lines):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    N_tuz = len(tuzs)
    cmap = ListedColormap(plt.get_cmap('viridis')(np.linspace(0, 1, N_tuz)))
    colors = [cmap(i) for i in range(N_tuz)]
    
    for hex in hexagonal_structure.values():
        x_hex, y_hex = hex["x_hex"], hex["y_hex"]
        ax.plot(x_hex, y_hex, '-', color=colors[hex["cluster_id"]])
    
    for i, tuz in tuzs.items(): 
        ax.plot(tuz["x"], tuz["y"], '+', color=colors[i])
    
    for injection_line in injection_lines.values():
        ax.plot(
            [injection_line["x_from"], injection_line["x_to"]], 
            [injection_line["y_from"], injection_line["y_to"]], 
            '--', color="blue"
        )
        
    for extraction_line in extraction_lines.values():
        ax.plot(
            [extraction_line["x_from"], extraction_line["x_to"]], 
            [extraction_line["y_from"], extraction_line["y_to"]], 
            '--', color="red"
        )
        
    plt.axis('off')
    return fig

def get_costs(tuzs, injection_lines, extraction_lines, cost_params):
    fixed_original, fixed_additional = 0, 0
    variable_injection, variable_injection_linear = 0, 0
    variable_extraction, variable_extraction_linear = 0, 0
    fixed_original += len(tuzs) * cost_params["fixed"]["original"]
    for tuz in tuzs.values():
        fixed_additional += len(tuz["extraction_points"])//20 * cost_params["fixed"]["additional"]
    for injection_line in injection_lines.values():
        distance = get_distance(
            injection_line["x_from"], 
            injection_line["y_from"], 
            injection_line["z_from"], 
            injection_line["x_to"], 
            injection_line["y_to"], 
            injection_line["z_to"], 
        )
        variable_injection += cost_params["variable_injection"]["fixed"]
        variable_injection_linear += cost_params["variable_injection"]["variable"] * distance
    for extraction_line in extraction_lines.values():
        distance = get_distance(
            extraction_line["x_from"], 
            extraction_line["y_from"], 
            extraction_line["z_from"], 
            extraction_line["x_to"], 
            extraction_line["y_to"], 
            extraction_line["z_to"], 
        )
        variable_extraction += cost_params["variable_extraction"]["fixed"]
        variable_extraction_linear += cost_params["variable_extraction"]["variable"] * distance
    return (
        fixed_original, fixed_additional, 
        variable_injection, variable_injection_linear, 
        variable_extraction, variable_extraction_linear
    )

def get_value(hexagonal_structure, value_params):
    """computes the value of the mine, expressed in $ / year

    Args:
        hexagonal_structure (Dict): geographical structure of the mine
        value_params (Dict): values parameter

    Returns:
        float: value of the mine
    """
    nb_hex = len(hexagonal_structure)
    value_well = value_params["well_flow"] * value_params["avg_concentration"] * value_params["uranium_value"] * 1e-3 * 24 * 365.25
    return nb_hex * value_well


# reporting 

def format_number(number):
    """
    Format a number with spaces as thousands separators.
    
    Args:
    number (int or float): The number to format.
    
    Returns:
    str: The formatted number as a string.
    """
    return f"{number:,}".replace(",", " ")

def print_summary(hexagons_to_draw, tuzs, costs, injection_lines, extraction_lines, value):
    summary_str = ""
    # geography
    n_hex_x = max(hexagon[0] for hexagon in hexagons_to_draw) - min(hexagon[0] for hexagon in hexagons_to_draw) + 1
    n_hex_y = max(hexagon[1] for hexagon in hexagons_to_draw) - min(hexagon[1] for hexagon in hexagons_to_draw) + 1
    summary_str+=f'{len(tuzs)} TUZs\n'
    summary_str+=f'Grid : {len(hexagons_to_draw)} cells ({n_hex_x} x {n_hex_y})\n'
    summary_str+='\n\n'
    
    summary_str+=f'• {round(np.mean([len(tuz["extraction_points"]) for tuz in tuzs.values()]), 1)} extraction points on average\n\n'
    summary_str+=f'• {round(np.mean([len(tuz["injection_points"]) for tuz in tuzs.values()]), 1)} injection points on average\n\n'
    summary_str+='\n\n'

    # costs
    (
        fixed_original, fixed_additional, 
        variable_injection, variable_injection_linear, 
        variable_extraction, variable_extraction_linear
    ) = costs
    total_cost = fixed_original + fixed_additional + variable_injection + variable_injection_linear + variable_extraction + variable_extraction_linear
    fixed_cost = fixed_original + fixed_additional
    variable_costs = total_cost - fixed_cost
    
    summary_str+=f'• Average TUZ cost : {format_number(round(total_cost / len(tuzs)))} €\n\n'
    summary_str+=f'   --> {round(100 * fixed_cost / total_cost)} % fixed costs\n\n'
    summary_str+=f'   --> {round(100 * variable_costs / total_cost)} % variable costs\n\n'
    summary_str+='\n\n'

    # avg distances 
    injection_distances, extraction_distances = [], []
    for injection_line in injection_lines.values():
        distance = get_distance(
            injection_line["x_from"], 
            injection_line["y_from"], 
            injection_line["z_from"], 
            injection_line["x_to"], 
            injection_line["y_to"], 
            injection_line["z_to"], 
        )
        injection_distances.append(distance)
    for extraction_line in extraction_lines.values():
        distance = get_distance(
            extraction_line["x_from"], 
            extraction_line["y_from"], 
            extraction_line["z_from"], 
            extraction_line["x_to"], 
            extraction_line["y_to"], 
            extraction_line["z_to"], 
        )
        extraction_distances.append(distance)
    summary_str+=f'• Average injection line length : {round(np.mean(injection_distances), 1)} m\n\n'
    summary_str+=f'• Average extraction line length : {round(np.mean(extraction_distances), 1)} m\n\n'
    summary_str+='\n'

    summary_str+='Value :\n\n'
    summary_str+=f'• Total : {format_number(round(value))} € per year\n\n'
    summary_str+=f'• {format_number(round(value / len(hexagons_to_draw)))} € per cell per year\n\n'
    
    return summary_str

def plot_tuz_sizes(tuzs, standards=[]):
    n_extractions = [len(tuz["extraction_points"]) for tuz in tuzs.values()]
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(n_extractions, edgecolor='black', align='left')
    for std_size in standards:
        ax.axvline(x=std_size, color='grey', linestyle='--')
    
    ax.set_xlabel('Number of extraction wells')
    ax.set_ylabel('Count')
    ax.set_title('TUZ sizes')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig


# capa tuning

def plot_stacked_costs(
    tuz_capas,
    fixed_original_costs,
    fixed_additional_costs,
    variable_injection_costs,
    variable_injection_linear_costs,
    variable_extraction_costs,
    variable_extraction_linear_costs,
    hexagons_to_draw,
    ):
    costs = np.array(
        [
            fixed_original_costs,
            fixed_additional_costs,
            variable_injection_costs,
            variable_injection_linear_costs,
            variable_extraction_costs,
            variable_extraction_linear_costs,
        ]
    )
    max_value = np.sum(costs, axis=0).max() / len(hexagons_to_draw)
    labels = [
        "Fixed Original Costs",
        "Fixed Additional Costs",
        "Variable Injection Costs",
        "Variable Injection Linear Costs",
        "Variable Extraction Costs",
        "Variable Extraction Linear Costs",
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    scaled_costs = [
        [cost / len(hexagons_to_draw) for cost in costs_i] for costs_i in costs
    ]
    ax.stackplot(tuz_capas, scaled_costs, labels=labels)
    ax.legend(loc="upper left", ncol=3)
    ax.set_ylim(0, 1.3 * max_value)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Costs")
    ax.set_title("Stacked Filled Graph of Costs")
    return fig

def plot_avg_distances(tuz_capas, distances_extraction_wells):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tuz_capas, distances_extraction_wells)
    ax.set_xlabel('Avg TUZ capacity (Number of extraction wells)')
    ax.set_ylabel('Distance')
    ax.set_title('Evolution of Avg distance to extraction well w.r.t TUZ capacity')
    return fig

def tune_tuz_capa(hexagons_to_draw, hexagonal_structure, vertices, capex_params, values_params,  min_capa=10, maxcapa=60):
    tuz_capas, values = [], []
    distances_extraction_wells = []
    fixed_original_costs, fixed_additional_costs = [], []
    variable_injection_costs, variable_injection_linear_costs = [], []
    variable_extraction_costs, variable_extraction_linear_costs = [], []

    for tuz_capa in tqdm(range(min_capa, maxcapa+1)):
        N_tuz = math.ceil(len(hexagons_to_draw) / tuz_capa)
        hexagonal_structure, tuzs = get_positions_tuz(hexagonal_structure, N_tuz)
        injection_lines, extraction_lines = get_lines(hexagonal_structure, vertices, tuzs)
        avg_distance_extraction = np.mean([
            get_distance(
                injection_line["x_from"], 
                injection_line["y_from"], 
                injection_line["z_from"], 
                injection_line["x_to"], 
                injection_line["y_to"], 
                injection_line["z_to"], 
            )
            for injection_line in injection_lines.values()
        ])
        costs = get_costs(tuzs, injection_lines, extraction_lines, capex_params)
        (
            fixed_original, fixed_additional, 
            variable_injection, variable_injection_linear, 
            variable_extraction, variable_extraction_linear
        ) = costs
        value = get_value(hexagonal_structure, values_params)
        tuz_capas.append(tuz_capa)
        values.append(value)
        fixed_original_costs.append(fixed_original)
        fixed_additional_costs.append(fixed_additional)
        variable_injection_costs.append(variable_injection)
        variable_injection_linear_costs.append(variable_injection_linear)
        variable_extraction_costs.append(variable_extraction)
        variable_extraction_linear_costs.append(variable_extraction_linear)
        distances_extraction_wells.append(avg_distance_extraction)

    return (
        tuz_capas, values, 
        distances_extraction_wells, 
        fixed_original_costs, fixed_additional_costs, 
        variable_injection_costs, variable_injection_linear_costs, 
        variable_extraction_costs, variable_extraction_linear_costs
    )

def results_tuning(hexagons_to_draw, elements):
    (
        tuz_capas, values,
        distances_extraction_wells, 
        fixed_original_costs, fixed_additional_costs, 
        variable_injection_costs, variable_injection_linear_costs, 
        variable_extraction_costs, variable_extraction_linear_costs
    ) = elements
    
    stacked_costs_fig = plot_stacked_costs(
        tuz_capas,
        fixed_original_costs,
        fixed_additional_costs,
        variable_injection_costs,
        variable_injection_linear_costs,
        variable_extraction_costs,
        variable_extraction_linear_costs,
        hexagons_to_draw,
    )
    avg_distances_fig = plot_avg_distances(tuz_capas, distances_extraction_wells)
    
    return stacked_costs_fig, avg_distances_fig
