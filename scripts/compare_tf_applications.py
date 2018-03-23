from matplotlib import pyplot as plt
from CHECLabPy.core.file_handling import DL1Reader


def main():
    files = dict(
        # no_tf="/Volumes/gct-jason/dynamicrange/Run17493_ped_dl1.h5",
        # old_tf="/Volumes/gct-jason/dynamicrange/Run17493_oldtf_dl1.h5",
        # small_steps_tf="/Volumes/gct-jason/dynamicrange/Run17493_smallsteps_dl1.h5",
        # tf_with_ped_included="/Volumes/gct-jason/dynamicrange/Run17493_withped_dl1.h5",
        # lei="/Volumes/gct-jason/dynamicrange/Run17493_lei_dl1.h5",
        # new="/Volumes/gct-jason/dynamicrange/Run17493_new_dl1.h5",
        ph="/Volumes/gct-jason/untitledfolder/Amplitude_12_dl1.h5",
        # p="/Volumes/gct-jason/dynamicrange/tf_comparisons/Run17493_withped_dl1.h5"
    )

    charges = dict()
    for key, fp in files.items():
        with DL1Reader(fp) as reader:
            pixel = reader.select_column("pixel").values
            asicch = pixel % 16
            charge = reader.select_column("charge").values
            charge = charge[asicch == 6]
            # charge /= charge.mean()
            charges[key] = charge

            # from IPython import embed
            # embed()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for key, charge in charges.items():
        ax.hist(charge, bins=100, range=[-2, 6], histtype='step', label=key)
    ax.legend(loc="upper right")

    plt.show()

    fig.savefig("tf_comparison.pdf", bbox_inches='tight')


if __name__ == '__main__':
    main()
