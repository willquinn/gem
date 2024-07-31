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
from dspeed import build_dsp
from lgdo import lh5
from scipy.optimize import curve_fit


usage = ""


import argparse
parser = argparse.ArgumentParser(description=usage)
parser.add_argument('-i', '--input', type=str, required=True,
                    help="Input PSS file")
parser.add_argument('-o', '--output', default=f"data/{GED_NAME}_pss_output_default.json", type=str,
                    help="Name of output file")
parser.add_argument('-c', '--dsp_config', default=f"dsp_proc_chain_emmulator.json", type=str,
                    help="Name of dsp_config file")
parser.add_argument('-v', '--verbosity', default=False, type=bool,
                    help="For debugging and printing out information")
parser.add_argument('-dsp', '--dsp_output', default=f"data/dsp_output.hdf", type=str,
                    help="Name of dsp_config file")
parser.add_argument('-plot', '--plot', default=False, type=bool,
                    help="Plot outputs")
parser.add_argument('-b', '--build', default=False, type=bool,
                    help="Plot outputs")

args = parser.parse_args()


def main():

    if args.build:
        build_dsp(
            args.input,
            args.dsp_output,
            json.load(open(args.dsp_config, "r")),
            write_mode="r",
            buffer_len=3200,
            block_width=16,
        )

    dsp_data = lh5.read_as("raw", args.dsp_output, 'pd')

    freq, bin_edges = np.histogram(
        dsp_data.trapEmax.values,
        range=(35750,36250), bins=500
    )
    bin_width = bin_edges[-1] - bin_edges[-2]
    bin_centres = bin_edges[:-1] + bin_width/2

    popt, pcov = curve_fit(
        f=gaus, xdata=bin_centres, ydata=freq,
        p0=[36100, 10, 100, 1]
    )
    initial_m = 2614.511/popt[0]

    peaks = np.array([583.191, 727.330, 860.564, 1592.511, 1620.50, 2103.511, 2614.511])

    popts = []
    pcovs = []
    perrs = []

    if args.plot:
        nrows, ncols = 3, 3
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,9))
    
    lowE, highE = 10, 10
    for index, peak in enumerate(peaks):
        i, j = index // nrows, index % ncols
        
        freq, bin_edges = np.histogram(
            dsp_data.query(f'(trapEmax > {(peak-15)/initial_m}) & (trapEmax < {(peak+15)/initial_m})').trapEmax.values*initial_m,
            range=(peak-lowE,peak+highE), bins=40
        )

        bin_width = bin_edges[-1] - bin_edges[-2]
        bin_centres = bin_edges[:-1] + bin_width/2

        if args.plot:
            axs[i][j].stairs(freq, bin_edges)
        
        popt, pcov = curve_fit(
            f=gaus, xdata=bin_centres, ydata=freq,
            p0=[peak, 10, 100, 1],
            bounds=[
                [peak-lowE, 0, 0, 0],
                [peak+highE, 20, 10000, 1000]
            ]
        )
        popts.append(popt)
        pcovs.append(pcov)
        perr = np.sqrt(np.diag(pcov))
        perrs.append(perr)
        
        if args.plot:
            axs[i][j].plot(bin_centres, gaus(bin_centres, *popt))
            axs[i][j].axvline(peak, ls='--', color='k', alpha=0.2)
            axs[i][j].text(
                peak-lowE, max(freq)*0.5,
                r"$\mu$ = "    + "({:.2f}".format(popt[0])+r"$\pm$"+"{:0.2f})\n".format(perr[0]) + 
                r"$\sigma$ = " + "({:.2f}".format(popt[1])+r"$\pm$"+"{:0.2f})\n".format(perr[1]) + 
                r"$A$ = "      + "({:.2f}".format(popt[2])+r"$\pm$"+"{:0.2f})\n".format(perr[2]) + 
                r"$h$ = "      + "({:.2f}".format(popt[3])+r"$\pm$"+"{:0.2f})\n".format(perr[3]) 
            )
            axs[i][j].set_title(f"{peak} keV")

    if args.plot:  
        fig.supxlabel('energy (keV)')
        fig.supylabel('counts /(0.5 keV)')
        plt.tight_layout()
        plt.savefig("plots/calibration_peaks.pdf")

    popt, pcov = curve_fit(
        f=linear, xdata=peaks, ydata=[h[0] for h in popts]/initial_m, sigma=np.array([np.sqrt(h[0,0]) for h in pcovs])/initial_m,
        p0=[initial_m**2, 0]
    )
    m_corr, c = popt[0], popt[1]
    trapEmax_cal = linear(dsp_data.trapEmax.values, 1/m_corr, -c/m_corr)
    dsp_data['trapEmax_cal'] = trapEmax_cal
    dsp_data['AoE'] = dsp_data.A_max.values/trapEmax_cal

    if args.plot:
        fig = plt.figure(figsize=(8, 6))
        gs = GridSpec(2, 1, height_ratios=[3, 1])

        ax1 = fig.add_subplot(gs[0])
        ax1.errorbar(peaks, [h[0] for h in popts]/initial_m, yerr=[np.sqrt(h[0,0]) for h in pcovs]/initial_m, fmt='.', label=r'fitted $\mu$')
        ax1.plot(peaks, linear(peaks, *popt), '-', label='linear fit')
        ax1.set_ylabel('trapEmax (ADC)')
        ax1.legend()
        ax1.set_xticklabels([])

        ax2 = fig.add_subplot(gs[1])
        ax2.errorbar(
            peaks, linear(peaks, *popt)-[h[0] for h in popts]/initial_m,
            yerr=[np.sqrt(h[0,0]) for h in pcovs]/initial_m, fmt=".",
        )
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.set_xlabel('energy (keV)')
        ax2.set_ylabel('Residuals')

        plt.subplots_adjust(hspace=0.1)
        plt.tight_layout()

        plt.savefig("plots/calibration_results.pdf")

        fig = plt.figure()
        plt.hist(
            dsp_data['trapEmax_cal'].values,
            range=(0, 4000), bins=2000, histtype='step'
        )
        plt.yscale('log')

        for i in peaks:
            plt.axvline(i, ls='--', color='k', alpha=0.2)
            
        plt.xlabel('energy calibrated (keV)')
        plt.ylabel('counts /(2 keV)')

        plt.savefig("plots/calibrated_espec.pdf")

        fig = plt.figure()
        plt.hist(
            dsp_data['AoE'].values,
            range=(0, 1), bins=2000, histtype='step'
        )
        plt.xlabel('A/E')
        plt.savefig("plots/aoe.pdf")

        fig = plt.figure()
        plt.scatter(
            dsp_data['trapEmax_cal'].values, dsp_data['AoE'].values,
            marker='.', alpha=0.2
        )

        plt.ylabel('A/E')
        plt.xlabel('energy (keV)')
        plt.savefig("plots/aoe_vs_energy.pdf")

    # filter the indices where A/E is not defined
    use_indices = np.where(~np.isnan(dsp_data['AoE'].values))[0]
    use_indices_df = pd.DataFrame(use_indices, columns=["use_indices"])
    use_indices_df.to_csv('data/not_nan_indices.csv', index=False)

    ys_df = pd.DataFrame({'Es': dsp_data['trapEmax_cal'], "AoEs": dsp_data['AoE'].values})
    ys_df.to_json(args.output, index=False)
    

if __name__ == "__main__":
    main()
