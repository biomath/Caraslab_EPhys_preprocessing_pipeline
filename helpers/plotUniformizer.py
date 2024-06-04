import matplotlib as mpl
import runpy
import seaborn as sns

if __name__ == "__main__":
    label_font_size = 11
    tick_label_size = 7
    legend_font_size = 6
    plot_line_thickness = 1
    axes_line_thickness = plot_line_thickness

    mpl.rcParams['figure.dpi'] = 600
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.weight'] = 'regular'
    mpl.rcParams['axes.labelweight'] = 'regular'

    mpl.rcParams['font.size'] = label_font_size
    mpl.rcParams['axes.labelsize'] = label_font_size
    mpl.rcParams['axes.titlesize'] = label_font_size
    mpl.rcParams['axes.linewidth'] = axes_line_thickness
    mpl.rcParams['legend.fontsize'] = legend_font_size
    mpl.rcParams['xtick.labelsize'] = tick_label_size
    mpl.rcParams['xtick.major.width'] = axes_line_thickness
    mpl.rcParams['ytick.labelsize'] = tick_label_size
    mpl.rcParams['ytick.major.width'] = axes_line_thickness
    mpl.rcParams['errorbar.capsize'] = label_font_size
    mpl.rcParams['lines.markersize'] = plot_line_thickness
    mpl.rcParams['lines.linewidth'] = plot_line_thickness

    mpl.rcParams['figure.figsize'] = (5, 5)
    runpy.run_module(mod_name='auROC_meanResponse_allUnitsCombined', run_name='__main__')
    # runpy.run_module(mod_name='auROC_meanResponse_rePlotClusters', run_name='__main__')
    # runpy.run_module(mod_name='auROC_meanResponse_groupedUnits_overplot', run_name='__main__')
    #
    # mpl.rcParams['figure.figsize'] = (10, 10)
    # runpy.run_module(mod_name='make_auROC_representativeFigures', run_name='__main__')