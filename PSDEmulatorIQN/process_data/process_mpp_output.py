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

        tree = uproot.open(f"{file_dir}{file}:simTree")
        df = tree.arrays(
            [
                'cluster_positionX', 'cluster_positionY', 'cluster_positionZ',
                'cluster_energy', 'mage_id', 'ievt', 'drift_times_vector'
            ], library='ak'
        )
        df['ievt_n'] = df['ievt'] + int(n_evts)

        df_sel = df[ak.any(df['mage_id'] == GED_MAGE_ID, axis=-1)]
        for i_df_sel in tqdm(df_sel):
            mage_id = i_df_sel.mage_id
            ievt_n = i_df_sel.ievt_n

            for index, mage_id_ in enumerate(mage_id):
                if mage_id_ == GED_MAGE_ID:
                    cx = i_df_sel.cluster_positionX[index]
                    cy = i_df_sel.cluster_positionY[index]
                    cz = i_df_sel.cluster_positionZ[index]
                    ce = i_df_sel.cluster_energy[index]*1000
                    sum_ce = sum(ce)
                    n_waveforms += 1

                    if sum_ce > 500: 
                        n_waveforms_pass += 1
                        for cx_, cy_, cz_, ce_ in zip(cx, cy, cz, ce):
                            cz_ = cz_ + GED_HEIGHT/2
                            cr_ = np.sqrt(cx_**2 + cy_**2)
                            dt_ = dt_file.GetTime(cr_, cz_)
                    
                            data_dict['E'].append(ce_),
                            data_dict['r'].append(cr_),
                            data_dict['z'].append(cz_),
                            data_dict['y'].append(cy_),
                            data_dict['x'].append(cx_),
                            data_dict['t'].append(1.4E7)
                            data_dict['dts'].append(dt_),
                            data_dict['Esum'].append(sum_ce)
                            data_dict['ievt_n'].append(ievt_n)
                            append_PET_file(args.pet_file, data_dict)


        n_evts += int(max(df.ievt))              
        if n_waveforms_pass > args.nevents:
            break

    df_data = pd.DataFrame(data_dict)
    df_data = df_data.explode(['x', 'y', 'z', 'r', 'E', 'dts', 'Esum', 'ievt_n']).reset_index().drop(columns=['index'])
    df_data.to_json(args.output)

    print(n_waveforms, n_waveforms_pass)


if __name__ == "__main__":
    main()
