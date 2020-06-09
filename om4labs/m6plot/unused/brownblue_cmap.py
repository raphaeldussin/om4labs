def brownblue_cmap():
    # The color map below is from the Light & Bartlein collection
    # which is tested for several different forms of colorblindness
    #
    # Reference:
    # A. Light & P.J. Bartlein, "The End of the Rainbow? Color Schemes for
    # Improved Data Graphics," Eos,Vol. 85, No. 40, 5 October 2004.
    # http://geog.uoregon.edu/datagraphics/EOS/Light-and-Bartlein.pdf

    lb_brownblue_values = numpy.array(
        [
            [144, 100, 44],
            [187, 120, 54],
            [225, 146, 65],
            [248, 184, 139],
            [244, 218, 200],
            [255, 255, 255],  # [241,244,245],
            [207, 226, 240],
            [160, 190, 225],
            [109, 153, 206],
            [70, 99, 174],
            [24, 79, 162],
        ]
    )

    lb_brownblue_values = lb_brownblue_values / 255.0
    lb_brownblue_values = lb_brownblue_values[::-1, :]

    import matplotlib

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "brownblue", lb_brownblue_values
    )
    cmap.set_bad("w")
    matplotlib.cm.register_cmap(cmap=cmap)
    return cmap
