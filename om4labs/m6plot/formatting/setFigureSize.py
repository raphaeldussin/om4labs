import matplotlib.pyplot as plt


def setFigureSize(
    aspect=None,
    verticalresolution=None,
    horiztonalresolution=None,
    npanels=1,
    debug=False,
):
    """
    Set the figure size based on vertical resolution and aspect ratio (tuple of W,H).
    """
    if (not horiztonalresolution is None) and (not verticalresolution is None):
        if aspect is None:
            aspect = [horiztonalresolution, verticalresolution]
        else:
            raise Exception(
                "Aspect-ratio and both h-/v- resolutions can not be specified together"
            )
    if aspect is None:
        aspect = {1: [16, 9], 2: [1, 1], 3: [7, 10]}[npanels]
    if (not horiztonalresolution is None) and (verticalresolution is None):
        verticalresolution = int(1.0 * aspect[1] / aspect[0] * horiztonalresolution)
    if verticalresolution is None:
        verticalresolution = {1: 576, 2: 720, 3: 1200}[npanels]
    width = int(1.0 * aspect[0] / aspect[1] * verticalresolution)  # First guess
    if debug:
        print("setFigureSize: first guess width =", width)
    width = width + (width % 2)  # Make even
    if debug:
        print("setFigureSize: corrected width =", width)
    if debug:
        print("setFigureSize: height =", verticalresolution)
    plt.figure(figsize=(width / 100.0, verticalresolution / 100.0))  # 100 dpi always?
    if npanels == 1:
        plt.gcf().subplots_adjust(
            left=0.08, right=0.99, wspace=0, bottom=0.09, top=0.9, hspace=0
        )
    elif npanels == 2:
        plt.gcf().subplots_adjust(
            left=0.11, right=0.94, wspace=0, bottom=0.09, top=0.9, hspace=0.15
        )
    elif npanels == 3:
        plt.gcf().subplots_adjust(
            left=0.11, right=0.94, wspace=0, bottom=0.05, top=0.93, hspace=0.15
        )
    elif npanels == 0:
        pass
    else:
        raise Exception("npanels out of range")
