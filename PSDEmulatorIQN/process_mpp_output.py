###################################################################
#
#                     LEGEND IQN PSD Emulator
#       github:
#       Intelectual property of:
#
#       process_mpp_output.py -> 
#
###################################################################

usage = """
This application processes clustered step information from mpp files. It
outputs two files, one for input into PSS processor and another for a
mpp summary that is quicker to load.
"""

from utils import *


import argparse
parser = argparse.ArgumentParser(description=usage)
parser.add_argument('-i', '--input', type=str, required=True,
                    help="Input MPP file(s)")
parser.add_argument('-o', '--output', default=f"data/{GED_NAME}_mpp_output_default.json", type=str,
                    help="Name of output file")
parser.add_argument('-pet', '--pet_file', default=f"data/{GED_NAME}_pet_default.csv", type=str,
                    help="Name of output file")
parser.add_argument('-n', '--nevents', default=30000, type=int,
                    help="Limit on the number of events to process. By default process all")
parser.add_argument('-v', '--verbosity', default=False, type=bool,
                    help="For debugging and printing out information")
parser.add_argument('-dt', '--drift_time_file', default=f"data/{GED_NAME}_drift_time_map.dat", type=str,
                    help="Drift time map for the detector of interest")
args = parser.parse_args()


def main():

    dt_file = ReadDTFile(args.drift_time_file)
    
    if '.root' in args.input:
        files = [args.input]
        file_dir = args.input.split(".root")[0]
    else:
        files = sorted(os.listdir(args.input))
        file_dir = args.input

    n_waveforms_pass = 0
    n_waveforms = 0
    n_evts = 0

    data_dict = {
        f'{i}': [] for i in ['x', 'y', 'z', 'r', 't', 'dts', 'E', 'Esum', 'ievt_n']
    }

    create_PET_file(args.pet_file)

    for ifile in tqdm(range(len(files))):
        file = files[ifile]
        if '.root' not in file:
            msg = f'> Input file must be .root format {file_dir}{file}'
            print(msg)
            continue
        # print(file, n_waveforms)

        tree = uproot.open(f"{file_dir}{file}:simTree")
        df = tree.arrays(
            [
                'cluster_positionX', 'cluster_positionY', 'cluster_positionZ',
                'cluster_energy', 'mage_id', 'ievt', 'drift_times_vector'
            ], library='ak'
        )
        df['ievt_n'] = df['ievt'] + int(n_evts)
        ievt_counts_df = ak.to_dataframe(df.ievt).value_counts()
        drop_evts = ievt_counts_df[ievt_counts_df>1].index.get_level_values(0).values

        for ievt in drop_evts:
            df = df[df.ievt != ievt]
        tmp_df = df[(ak.num(df.mage_id) == 1)]
        df_mage_id = tmp_df[ak.flatten(tmp_df.mage_id == GED_MAGE_ID)]

        assert len(df_mage_id) < len(df)
                    
        df_mage_id['cluster_energy'] = df_mage_id['cluster_energy']*1000
        df_mage_id['cluster_positionR'] = np.sqrt(df_mage_id['cluster_positionX']**2 + df_mage_id['cluster_positionY']**2)
        df_mage_id['cluster_positionZ'] = df_mage_id['cluster_positionZ'] + 111.8/2
        df_mage_id['cluster_positionT'] = df_mage_id['cluster_positionX']*0 + 1.4E7
        df_mage_id['drift_times'] = nested_drift_time(dt_file.GetTime, df_mage_id['cluster_positionR'], df_mage_id['cluster_positionZ'])
        df_mage_id['cluster_energy_sum'] = ak.sum(df_mage_id['cluster_energy'], axis=-1)

        for ievt_index in tqdm(range(len(df_mage_id.ievt))):
            ievt = df_mage_id.ievt[ievt_index]
            df_temp = df_mage_id[df_mage_id.ievt == ievt]
                
            assert len(df_temp.cluster_energy.to_numpy()) == 1
            assert len(df_temp.cluster_energy.to_numpy()[0]) == 1

            xs = df_temp.cluster_positionX.to_numpy()[0][0]
            ys = df_temp.cluster_positionY.to_numpy()[0][0]
            zs = df_temp.cluster_positionZ.to_numpy()[0][0]
            rs = df_temp.cluster_positionR.to_numpy()[0][0]
            ts = df_temp.cluster_positionT.to_numpy()[0][0]
            Es = df_temp.cluster_energy.to_numpy()[0][0]
            ns = df_temp.ievt_n.to_numpy()[0]
            dts = df_temp.drift_times.to_numpy()[0][0]
            sumE = np.ones_like(Es)*df_temp.cluster_energy_sum.to_numpy()[0]
                    
            assert sum(Es) == sumE[0]

            n_waveforms += 1
            do_waveform = True
                    
            if sum(Es) < 500: do_waveform = False
            if do_waveform:
                n_waveforms_pass += 1
                data_dict['x'].append(xs)
                data_dict['y'].append(ys)
                data_dict['z'].append(zs)
                data_dict['r'].append(rs)
                data_dict['t'].append(ts)
                data_dict['E'].append(Es)
                data_dict['dts'].append(dts)
                data_dict['Esum'].append(sumE)
                data_dict['ievt_n'].append(ns*np.ones_like(Es))

                append_PET_file(args.pet_file, data_dict)

        n_evts += int(max(df.ievt))              
        # print(n_waveforms_pass, n_evts)
        if n_waveforms_pass > args.nevents:
            break

    df_data = pd.DataFrame(data_dict)
    df_data = df_data.explode(['x', 'y', 'z', 'r', 'E', 'dts', 'Esum', 'ievt_n']).reset_index().drop(columns=['index'])
    df_data.to_json(args.output)


if __name__ == "__main__":
    main()
