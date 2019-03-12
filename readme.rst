Description
===========

TEdgeMask is an edge detection filter.

This is a port of the TEdgeMask/TEMmod Avisynth plugins.


Usage
=====
::

    tedgemask.TEdgeMask(clip clip, [float[] threshold=8.0, int type=2, int link, float scale=1.0, int[] planes=<all>, bint opt=True])


Parameters:
    *clip*
        A clip to process. It must have constant format, 8..16 bit
        integer sample type, and subsampling ratios of at most 2.

    *threshold*
        Sets the magnitude thresholds.

        If over this value then a sample will be considered an edge,
        and the output pixel will be set to the maximum value allowed
        by the format. Otherwise the output pixel will be set to 0.

        Set this to 0 to output a magnitude mask instead of a binary
        mask.

        Default: 8.0 for the first plane, and the previous plane's
        threshold for the other planes.

    *type*
        Sets the type of first order partial derivative approximation
        that is used.

        1 - 2 pixel.

        2 - 4 pixel.

        3 - Same as *type* = 1.

        4 - Same as *type* = 2.

        5 - 6 pixel (Sobel operator).

        Default: 2.

    *link*
        Specifies whether luma to chroma linking, no linking, or
        linking of every plane to every other plane is used.

        0 - No linking. The three edge masks are completely
        independent.

        1 - Luma to chroma linking. If a luma pixel is considered an
        edge, the corresponding chroma pixel is also marked as an
        edge.

        2 - Every plane to every other plane. If a pixel is considered
        an edge in any plane, the corresponding pixels in all the
        other planes are also marked as edges.

        This parameter has no effect when *clip* is GRAY, when any
        plane's *threshold* is 0, or when some planes are not
        processed.

        This parameter can only be 0 or 2 when *clip* is RGB.

        Default: 2 when *clip* is RGB, otherwise 1.

    *scale*
        If the output is a magnitude mask (*threshold* is 0), it is
        scaled by this value.

        Note that in TEMmod this parameter had three different,
        undocumented default values for the different mask types,
        which made it difficult to use the parameter without reading
        the source code.

        Default: 1.0.

    *planes*
        Select which planes to process. Any unprocessed planes will be
        copied.

        Default: all the planes.

    *opt*
        If True, the best optimised functions supported by the CPU
        will be used. If False, only scalar functions will be used.

        Default: True.


Compilation
===========

::

    meson build && cd build
    ninja


License
=======

GNU GPL v2, like the Avisynth plugins.
