import numpy as np


def generate_random_condition(min_element_num=None, max_element_num=None, dimensions=None, distributed=True, multi_load=True, n_material=1, max_vf=0.75, min_vf=0.15, min_vf_per_material=0.02):
    if dimensions:
        dims = dimensions
    elif min_element_num and max_element_num:
        element_num = np.random.uniform(min_element_num, max_element_num)
        ar1 = np.exp(np.random.normal(0, 1, 1))
        a = (element_num*ar1)**(1/2)
        b = a/ar1
        dims = (a, b)
        dims = np.rint(dims).astype(int).flatten()
    else:
        raise Exception("Either dimensions or min_element_num and max_element_num must be specified.")
    
    if multi_load:
        num_loads = np.random.geometric(0.3)
    else:
        num_loads = 1
    num_constraints = np.random.geometric(0.2) + 1

    x_fixtures = []
    y_fixtures = []

    all_load_pos = []
    all_load_vec = []
    for load in range(num_loads):
        #Calcualte the load direction and magnitude
        load_dir = np.random.choice(["x","y","xy"], p=[0.2, 0.2, 0.6])
        
        load_options_dict = {
                        "edge_point": 0.1,
                        "full_edge": 0.1*distributed*multi_load,
                        "partial_edge": 0.1*distributed*multi_load,
                        "corner": 0.1,
                        "internal_distributed_orthogonal": 0.025*distributed*multi_load,
                        "internal_distributed": 0.025*distributed*multi_load,
                        "internal_distributed_ellipse": 0.025*distributed*multi_load,
                        "internal_distributed_rectangle": 0.025*distributed*multi_load,
                        "internal_point": 0.50,
        }
        
        load_pos_s, load_vec_s= get_load_mag_and_vec(dims, load_options_dict, load_dir)
        for i in range(len(load_pos_s)):
            load_pos = tuple(load_pos_s[i,:])
            load_vec = tuple(load_vec_s[i,:])
            if load_pos not in all_load_pos: #if duplicate, try again
                all_load_pos.append(load_pos)
                all_load_vec.append(load_vec)
        
    all_load_pos = np.array(all_load_pos)
    all_load_vec = np.array(all_load_vec)

    constraint_dirs = []
    for constraint in range(num_constraints):
        constraint_dir = np.random.choice(["x","y","xy"], p=[0.3, 0.3, 0.4])
        constraint_dirs.append(constraint_dir)
    
    constraint_options_dict = {
                        "edge_point": 0.1,
                        "full_edge": 0.1*distributed,
                        "partial_edge": 0.1*distributed,
                        "corner": 0.1,
                        "internal_distributed_orthogonal": 0.025*distributed,
                        "internal_distributed": 0.025*distributed,
                        "internal_distributed_ellipse": 0.025*distributed,
                        "internal_distributed_rectangle": 0.025*distributed,
                        "internal_point": 0.50,
        }
    for constraint_dir in constraint_dirs:
        constraint_pos_s = get_constraints(dims, constraint_options_dict)
        for i in range(len(constraint_pos_s)):
            constraint_pos = tuple(constraint_pos_s[i,:])
            constraint_pos = tuple(constraint_pos)
            if constraint_dir == "x":
                x_fixtures.append(constraint_pos)
            elif constraint_dir == "y":
                y_fixtures.append(constraint_pos)
            elif constraint_dir == "xy":
                x_fixtures.append(constraint_pos)
                y_fixtures.append(constraint_pos)
    #eliminate duplicate constraints
    x_fixtures = list(set(x_fixtures))
    y_fixtures = list(set(y_fixtures))
    x_fixtures = np.array(x_fixtures)
    y_fixtures = np.array(y_fixtures)

    #Check if fixtures fully constrain the system, if not regenerate with recursion
    a, b = len(x_fixtures), len(y_fixtures)
    lowest, highest = sorted([a, b])[0], sorted([a, b])[1]
    if lowest<1 or highest<2:
        return generate_random_condition(min_element_num=min_element_num, max_element_num=max_element_num, dimensions=dimensions, distributed=distributed, multi_load=multi_load, n_material=n_material, max_vf=max_vf, min_vf=min_vf, min_vf_per_material=min_vf_per_material)
    all_loads_in_constraints=True
    for load in all_load_pos:
        if load in x_fixtures or load in y_fixtures:
            pass
        else:
            all_loads_in_constraints=False
    if all_loads_in_constraints:
        return generate_random_condition(min_element_num=min_element_num, max_element_num=max_element_num, dimensions=dimensions, distributed=distributed, multi_load=multi_load, n_material=n_material, max_vf=max_vf, min_vf=min_vf, min_vf_per_material=min_vf_per_material)
    
    # #Scale
    # max_dim = max(dims)-1 #Subtract one to calculate max number of elements (rather than nodes)
    # all_load_pos = [p/max_dim for p in all_load_pos]
    # x_fixtures = [x/max_dim for x in x_fixtures]
    # y_fixtures = [y/max_dim for y in y_fixtures]

    # normalize loads by max load magnitude
    max_load = np.max(np.linalg.norm(all_load_vec, axis=1))
    all_load_vec = all_load_vec / max_load
    
    if n_material > 1:
        n_material = np.random.randint(2, n_material + 1)
    
    E_mat = np.random.uniform(size=n_material)
    E_mat /= E_mat.max()
    E_mat = E_mat.reshape(-1, 1)

    total_vf = np.random.uniform(min_vf, max_vf)
    vf = np.random.uniform(size=n_material)
    vf /= vf.sum()
    vf *= total_vf

    while np.any(vf < min_vf_per_material):
        vf = np.random.uniform(size=n_material)
        vf /= vf.sum()
        vf *= total_vf

    vf = vf.reshape(-1, 1)

    all_fixtures, fixtures_directions = find_duplicates_in_two_arrays(x_fixtures, y_fixtures)

    return dims, all_load_pos, all_load_vec, all_fixtures, fixtures_directions, E_mat, vf


def find_duplicates_in_two_arrays(arr1, arr2):
    # Combine the two arrays
    combined_arr = np.vstack((arr1, arr2))

    # Find the unique rows and the indices of the duplicates
    unique_arr, indices, inverse_indices, counts = np.unique(combined_arr, axis=0, return_index=True, return_inverse=True, return_counts=True)
    
    # Find the duplicate row indices
    duplicate_indices = np.where(counts > 1)[0]
    duplicate_rows = [np.where(inverse_indices == index)[0] for index in duplicate_indices]

    # Deduplicated array
    deduplicated_array = unique_arr

    # Source array: each row indicates which arrays (arr1 or arr2) the rows in the deduplicated array came from
    source_array = np.zeros((len(unique_arr), 2), dtype=int)
    for idx, row in enumerate(unique_arr):
        # Check if the row is in arr1
        if np.any(np.all(arr1 == row, axis=1)):
            source_array[idx, 0] = 1
        # Check if the row is in arr2
        if np.any(np.all(arr2 == row, axis=1)):
            source_array[idx, 1] = 1
    
    return deduplicated_array, source_array

def get_load_mag_and_vec(dims, load_options_dict, load_dir):
    #Calculate and normalize load probabilities and options
    load_options = list(load_options_dict.keys())
    load_probs = [load_options_dict[key] for key in load_options]
    load_probs = np.array(load_probs)
    load_probs = load_probs / np.sum(load_probs)

    #Calculate the load magnitude
    load_mag = np.random.exponential(1)
    #Calculate the load position
    load_pos = random_load_or_cond(dims, load_options, load_probs)
    #Calculate the load vector
    load_vec = random_vector(load_mag, load_dir)
    num_loads = len(load_pos)
    load_vec = np.array([load_vec/num_loads]*num_loads)
    return load_pos, load_vec


def get_constraints(dims, constraint_options_dict, face=None):
    #Calculate and normalize constraint probabilities and options
    constraint_options = list(constraint_options_dict.keys())
    constraint_probs = [constraint_options_dict[key] for key in constraint_options]
    constraint_probs = np.array(constraint_probs)
    constraint_probs = constraint_probs / np.sum(constraint_probs)

    #calculate constraint 
    constraint_pos = random_load_or_cond(dims, constraint_options, constraint_probs, face=face)
    return constraint_pos

def random_load_or_cond(dims, options, probs, face=None):
    type = np.random.choice(options, p=probs)
    if type == "internal_point":
        x = np.random.choice(range(0, dims[0]-1))
        y = np.random.choice(range(0, dims[1]-1))
        load_pos = [(x, y)]
    elif type == "face":
        if face:
            pass
        else:
            face = np.random.choice(["x", "y"])
        if face == "x":
            x = np.random.choice([0, dims[0]-1])
            y = np.random.choice(range(0, dims[1]-1))
        elif face == "y":
            x = np.random.choice(range(0, dims[0]-1))
            y = np.random.choice([0, dims[1]-1])
        load_pos = [(x, y)]
    elif type == "edge_point":
        edge = np.random.choice(["x","y"])
        if edge == "x":
            x = np.random.choice(range(dims[0]))
            y = np.random.choice([0, dims[1]-1])
        elif edge == "y":
            x = np.random.choice([0, dims[0]-1])
            y = np.random.choice([0, dims[1]-1])
        load_pos = [(x, y)]
    elif type == "full_edge":
        #distributed load across the entire edge
        edge = np.random.choice(["x","y"])
        load_pos = []
        if edge == "x":
            y = np.random.choice([0, dims[1]-1])
            for x in range(dims[0]):
                load_pos.append((x, y))
        elif edge == "y":
            x = np.random.choice([0, dims[0]-1])
            for y in range(dims[1]):
                load_pos.append((x, y))
    elif type == "partial_edge":
        edge = np.random.choice(["x","y"])
        load_pos = []
        if edge == "x":
            y = np.random.choice([0, dims[1]-1])
            xlen = np.random.choice(range(1,dims[0]//2))
            xc = np.random.choice(range(dims[0]))
            xl = xc-xlen
            xr = xc+xlen
            xl = max(0, xl)
            xr = min(dims[0]-1, xr)
            for x in range(xl, xr):
                load_pos.append((x, y))
        elif edge == "y":
            x = np.random.choice([0, dims[0]-1])
            ylen = np.random.choice(range(1,dims[1]//2))
            yc = np.random.choice(range(dims[1]))
            yl = yc-ylen
            yr = yc+ylen
            yl = max(0, yl)
            yr = min(dims[1]-1, yr)
            for y in range(yl, yr):
                load_pos.append((x, y))
    elif type == "internal_distributed_orthogonal":
        edge = np.random.choice(["x","y"])
        load_pos = []
        if edge == "x":
            y = np.random.choice(range(dims[1]))
            xlen = np.random.choice(range(1,dims[0]//2))
            xc = np.random.choice(range(dims[0]))
            xl = xc-xlen
            xr = xc+xlen
            xl = max(0, xl)
            xr = min(dims[0]-1, xr)
            for x in range(xl, xr):
                load_pos.append((x, y))
        elif edge == "y":
            x = np.random.choice(range(dims[0]))
            ylen = np.random.choice(range(1,dims[1]//2))
            yc = np.random.choice(range(dims[1]))
            yl = yc-ylen
            yr = yc+ylen
            yl = max(0, yl)
            yr = min(dims[1]-1, yr)
            for y in range(yl, yr):
                load_pos.append((x, y))
    elif type == "internal_distributed":
        xc = np.random.choice(range(dims[0]))
        yc = np.random.choice(range(dims[1]))
        xlen = np.random.choice(range(1,dims[0]//2))
        ylen = np.random.choice(range(1,dims[1]//2))
        xl = xc-xlen
        xr = xc+xlen
        yl = yc-ylen
        yr = yc+ylen
        xl = max(0, xl)
        xr = min(dims[0]-1, xr)
        yl = max(0, yl)
        yr = min(dims[1]-1, yr)
        load_pos = []
        if xlen>=ylen:
            for x in range(xl, xr):
                y = (x-xl)/(xr-xl)*(yr-yl)+yl
                y = int(y)
                load_pos.append((x, y))
        else:
            for y in range(yl, yr):
                x = (y-yl)/(yr-yl)*(xr-xl)+xl
                x = int(x)
                load_pos.append((x, y))
    elif type == "internal_distributed_ellipse":
        xc = np.random.choice(range(dims[0]))
        yc = np.random.choice(range(dims[1]))
        xp = min(20/float(dims[0]), 1)
        yp = min(20/float(dims[1]), 1)
        xlen = np.random.geometric(xp)
        ylen = np.random.geometric(yp)
        xlen = int(xlen)
        ylen = int(ylen)
        xl = xc-xlen
        xr = xc+xlen
        yl = yc-ylen
        yr = yc+ylen
        angle = np.random.uniform(0, 2*np.pi)
        load_pos = []
        for x in range(dims[0]):
            for y in range(dims[1]):
                in_ellipse = ((x-xc)*np.cos(angle)+(y-yc)*np.sin(angle))**2/xlen**2 + ((x-xc)*np.sin(angle)-(y-yc)*np.cos(angle))**2/ylen**2 <= 1
                if in_ellipse:
                    load_pos.append((x, y))
    elif type == "internal_distributed_rectangle":
        xc = np.random.choice(range(dims[0]))
        yc = np.random.choice(range(dims[1]))
        xp = min(20/float(dims[0]), 1)
        yp = min(20/float(dims[1]), 1)
        xlen = np.random.geometric(xp)
        ylen = np.random.geometric(yp)
        xlen = int(xlen)
        ylen = int(ylen)
        xl = xc-xlen
        xr = xc+xlen
        yl = yc-ylen
        yr = yc+ylen
        load_pos = []
        for x in range(xl, xr):
            for y in range(yl, yr):
                load_pos.append((x, y))
    elif type == "corner":
        x = np.random.choice([0, dims[0]-1])
        y = np.random.choice([0, dims[1]-1])
        load_pos = [(x, y)]
    load_pos = np.array(load_pos)
    return load_pos

def random_vector(mag, type):
    if type == "x":
        load_vec = (mag, 0)
    elif type == "y":
        load_vec = (0, mag)
    elif type == "xy":
        angle = np.random.uniform(0, 2*np.pi)
        load_vec = (mag*np.cos(angle), mag*np.sin(angle))
    load_vec = np.array(load_vec)
    return load_vec