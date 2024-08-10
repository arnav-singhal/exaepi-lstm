aggregate_column = b'FIPS'
# aggregate_column = b'Tract'
NUM_DAYS = 100
DATA_DIR = "/global/cfs/cdirs/m3623/lstm_data/"

import numpy as np
import pandas as pd
from mpi4py import MPI
import h5py
import glob

def get_comms_from_pos(xs: float, ys: float):
    """Get community index from AMReX position values.
    These are the first two components in the real data
    for every particle."""
    
    x_max = np.rint(np.max(1 / xs) / 2)
    y_max = np.rint(np.max(1 / ys) / 2)
    
    adj_xs = np.rint(x_max * xs - 0.5)
    adj_ys = np.rint(y_max * ys - 0.5)
    return (adj_xs + adj_ys * x_max).astype(int)

def get_comms_from_coord(i: int, j: int):
    """Get community index from home/work_i and _j coordinates.
    These data are only written on Day 0, so must be used with
    Day 0 data files."""
    
    x_max = np.max(i) + 1
    return i + j * x_max

def comms_to_counties(comms, comm_ref, county_ref):
    """Converts community index to county FIPS code.
    Inputs:
        comms: array of community indices to be converted
        comm_ref: array of all community indices, in same order as county_ref
        county_ref: array of all FIPS codes, in same order as comm_ref
    Output: array of same length as comms, with community indices
            replaced by respective FIPS codes"""
    
    max_comm = np.max(comm_ref) + 1
    ordered_counties = np.zeros(max_comm, dtype=int)
    for idx, comm in np.ndenumerate(comm_ref):
        if comm != -1:
            ordered_counties[comm] = county_ref[idx]
    return ordered_counties[comms]

def fast_get_counties(data_dir : str):
    """Get home counties for each particle
    Outputs: nd.nparray[int] (list of counties for each particle),
             nd.nparray[int] (list of sorted, unique counties)"""
    
    if data_dir[-1] != "/":
        data_dir = data_dir + "/"
        
    # Day 1 data is used since it is smaller and takes less load time than Day 0
    with h5py.File(data_dir + "plt00001/agents/agents.h5", 'r') as agent_file, \
        h5py.File(data_dir + "plt00001.h5", 'r') as comm_file:
    
        # Load all particle data
        # int_data = agent_file['level_0']['data:datatype=0'][()] \
        #     .reshape(-1, agent_file.attrs['num_component_int'][0] + 2)
        real_data = agent_file['level_0']['data:datatype=1'][()] \
            .reshape(-1, agent_file.attrs['num_component_real'][0] + 2)

        # Determine indices of components
        comm_indices = {}
        for i in range(comm_file.attrs['num_components'][0]):
            comm_indices[comm_file.attrs['component_' + str(i)]] = str(i)
        
        counties = np.rint(comm_file['level_0']['data:datatype=' + comm_indices[aggregate_column]][()]).astype(int)
        sorted_counties = np.unique(counties)
        # if garbage communities exist, will have a county of -1
        sorted_counties = sorted_counties[1:] if sorted_counties[0] == -1 else sorted_counties
        
        # 0th and 1st index of real data are always x- and y-positions
        return comms_to_counties(get_comms_from_pos(real_data[:, 0], real_data[:, 1]),
                                 np.rint(comm_file['level_0']['data:datatype=' + comm_indices[b'comm']][()]).astype(int),
                                 counties), \
            sorted_counties

def get_withdrawn(data_dir: str = "sim/180_run", days: int = 100):
    """An example function that aggregates infection counts across counties and age group.
    Can be customized by changing what field is being matched, or by changing
    get_counties to match census tract instead of FIPS code, etc.
    Inputs: data_dir: data directory
            days: number of days to aggregate
    Output: (days, 5, # of counties)-array with summed counts.
    """
    data_dir = data_dir[:-1] if data_dir[-1] == "/" else data_dir

    # Get counties of each particle
    particle_counties, sorted_counties = fast_get_counties(data_dir)
    
    # Set up output array: day x age group x county
    out = np.zeros((days, 5, len(sorted_counties)), dtype=np.float32)

    for day in range(days):
        with h5py.File(f"{data_dir}/plt{day:05}/agents/agents.h5", 'r') as agent_file:
            # print(f"Starting Day {day}")
            int_indices = {}
            for i in range(2, agent_file.attrs['num_component_int'][0] + 2):
                int_indices[agent_file.attrs['int_component_' + str(i)]] = i

            int_data = agent_file['level_0']['data:datatype=0'][()] \
                .reshape(-1, agent_file.attrs['num_component_int'][0] + 2)
            
            if day == 0:
                # Generate masks for each age group
                ages = int_data[:, int_indices[b'age_group']]
                agerange = np.arange(5).reshape(5, 1)
                age_masks = ages == agerange

            # print("Aggregating\n")
            for age in range(5):
                # Get an array of counties that match status == 1 and the corresponding age group
                infected_counties = particle_counties[(int_data[:, int_indices[b'withdrawn']] == 1) & age_masks[age]]
                
                # Count up number of each county
                counts = pd.Series(infected_counties).value_counts()
                
                # Find index for the counted counties in the sorted_counties order
                # This makes sure if any counties are missing that all counties are slotted
                # correctly.
                present_counties = np.searchsorted(sorted_counties, counts.index.values)
                
                # Set corresponding slice of output array to the counts
                out[day, age, present_counties] = counts.values
    return out

def write_all_seeds(days: int, num_age_groups: int = 5, num_counties: int = 9):
    """Write each seed's age/county counts to an HDF5 file, using MPI
    to parallelize."""
    
    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    rank = comm.Get_rank()
    
    # We assume every seed's data is in a folder named seed[xxx].
    seeds = glob.glob(DATA_DIR + "seed*")
    seed_size = len(seeds)
    block_size = np.ceil(seed_size / mpi_size).astype(int)
    
    if aggregate_column == b'FIPS':
        data_prefix = "county"
    elif aggregate_column == b'Tract':
        data_prefix = "tract"
    else:
        data_prefix = "unknown"
    with h5py.File(f"{aggregate_column}_data.h5", "a", libver="latest", driver="mpio", comm=comm) as f:
        ds = f.require_dataset("withdrawn", (seed_size, days, 5, 9), "<f4")
    
        for i in range(block_size):
            seed_idx = rank * block_size + i
            if seed_idx >= seed_size:
                break
            # print(f"Rank {rank} / {size}: doing {seeds[seed_idx]}, {i}/{block_size}", flush=True)
            ds[seed_idx] = get_withdrawn(seeds[seed_idx], days)
            # print(f"Rank {rank} / {size}: done", flush=True)
            if i % 100 == 0:
                print(f"Rank {rank}: done {i}")

if __name__ == "__main__":
    write_all_seeds(NUM_DAYS)