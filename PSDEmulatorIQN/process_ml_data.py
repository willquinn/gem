###################################################################
#
#                     LEGEND IQN PSD Emulator
#       github:
#       Intelectual property of:
#
#       process_pss_output.py -> 
#
###################################################################


from utils import *


usage = ''


import argparse
parser = argparse.ArgumentParser(description=usage)
parser.add_argument('-pss', '--pss_input', default=f"data/{GED_NAME}_pss_output_default.json", type=str,
                    help="Input PSS file")
parser.add_argument('-mpp', '--mpp_input', default=f"data/{GED_NAME}_mpp_output_default.json", type=str,
                    help="Name of output file")
parser.add_argument('-o', '--output', default=f"data/ml_data.json", type=str,
                    help="Name of dsp_config file")
parser.add_argument('-v', '--verbosity', default=False, type=bool,
                    help="For debugging and printing out information")

args = parser.parse_args()


def main():

    mpp_data = pd.read_json(args.mpp_input)
    pss_data = pd.read_json(args.pss_input)

    n_clusters = 4
    data_dict = {}

    for i in range(n_clusters):
        data_dict[f"E{i}"] = []
        data_dict[f"dt{i}"] = []
    data_dict["Esum"] = []
    data_dict["ievt_n"] = []

    grouped = mpp_data.groupby('ievt_n')

    new_indexes = pd.read_csv('data/new_index.csv').Column1.values.astype('int')

    # Iterate through each group
    for ievt_n, group in grouped:
        if ievt_n in new_indexes: continue
        
        # Get the cluster_energy values for the group
        energies = group['E'].values
        dts = group['dts'].values
        
        # Sort energies in descending order and get the indices
        sorted_indices = np.argsort(energies)[::-1]
        for i in range(n_clusters):
            clust_index = sorted_indices[i] if len(sorted_indices) > i else None
            E_clust = energies[clust_index] if clust_index is not None else 0
            dt_clust = dts[clust_index] if clust_index is not None else 0
            data_dict[f"E{i}"].append(E_clust)
            data_dict[f"dt{i}"].append(dt_clust)
        
        data_dict["Esum"].append(sum(energies))
        data_dict["ievt_n"].append(ievt_n)
        
    for key, item in data_dict.items():
        data_dict[key] = np.array(item)
    df_mpp_data = pd.DataFrame(data_dict)

    df_data = pd.concat([pss_data, df_mpp_data], axis=1)

    df_data.to_json(args.output)

    

if __name__ == "__main__":
    main()
