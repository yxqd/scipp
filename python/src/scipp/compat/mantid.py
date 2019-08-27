import scipp as sc
import numpy as np


def get_pos(pos):
    return [pos.X(), pos.Y(), pos.Z()]


def convert_instrument(ws):
    compInfo = sc.Dataset(
        {'position': sc.Variable(dims=[sc.Dim.Row], shape=(2,),
                                 dtype=sc.dtype.vector_3_double,
                                 unit=sc.units.m)})
    # Current assumption: 0 is source, 1 is sample
    sourcePos = ws.componentInfo().sourcePosition()
    samplePos = ws.componentInfo().samplePosition()
    compInfo['position'].values[0] = get_pos(sourcePos)
    compInfo['position'].values[1] = get_pos(samplePos)
    return sc.Variable(value=compInfo)


def initPosSpectrumNo(nHist, ws):

    pos = np.zeros([nHist, 3])
    num = np.zeros([nHist], dtype=np.int32)

    spec_info = ws.spectrumInfo()
    for i in range(nHist):
        p = spec_info.position(i)
        pos[i, :] = [p.X(), p.Y(), p.Z()]
        # For V20, geometry x and y are swapped:
        # pos[i, :] = [p.Y(), p.X(), p.Z()]
        num[i] = ws.getSpectrum(i).getSpectrumNo()
    pos = sc.Variable([sc.Dim.Position], values=pos, unit=sc.units.m)
    num = sc.Variable([sc.Dim.Position], values=num)
    return pos, num


def ConvertWorkspace2DToDataset(ws):
    cb = ws.isCommonBins()
    nHist = ws.getNumberHistograms()
    comp_info = convert_instrument(ws)
    pos, num = initPosSpectrumNo(nHist, ws)

    # TODO More cases?
    allowed_units = {"DeltaE": sc.Dim.EnergyTransfer, "TOF": sc.Dim.Tof}
    xunit = ws.getAxis(0).getUnit().unitID()
    if xunit not in allowed_units.keys():
        raise RuntimeError("X-axis unit not currently supported for "
                           "Workspace2D. Possible values are: {}, "
                           "got '{}'. ".format(
                               [k for k in allowed_units.keys()], xunit))
    else:
        dim = allowed_units[xunit]

    if cb:
        coords = sc.Variable([dim], values=ws.readX(0))
    else:
        coords = sc.Variable([sc.Dim.Position, dim],
                             shape=(ws.getNumberHistograms(),
                                    len(ws.readX(0))))
        for i in range(ws.getNumberHistograms()):
            coords[sc.Dim.Position, i].values = ws.readX(i)

    # TODO Use unit information in workspace, if available.
    var = sc.Variable([sc.Dim.Position, dim],
                      shape=(ws.getNumberHistograms(), len(ws.readY(0))),
                      unit=sc.units.counts)

    for i in range(ws.getNumberHistograms()):
        var[sc.Dim.Position, i].values = ws.readY(i)
        var[sc.Dim.Position, i].variances = np.power(ws.readE(i), 2)

    return sc.DataArray(data=var, coords={dim: coords, sc.Dim.Position: pos},
                        labels={"spectrum_number": num,
                                "component_info": comp_info})


def ConvertEventWorkspaceToDataset(ws, drop_pulse_times):

    allowed_units = {"TOF": sc.Dim.Tof}
    xunit = ws.getAxis(0).getUnit().unitID()
    if xunit not in allowed_units.keys():
        raise RuntimeError("X-axis unit not currently supported for "
                           "EventWorkspace. Possible values are: {}, "
                           "got '{}'. ".format(
                               [k for k in allowed_units.keys()], xunit))
    else:
        dim = allowed_units[xunit]

    nHist = ws.getNumberHistograms()
    comp_info = convert_instrument(ws)
    pos, num = initPosSpectrumNo(nHist, ws)

    # TODO Use unit information in workspace, if available.
    coords = sc.Variable([sc.Dim.Position, dim],
                         shape=[nHist, sc.Dimensions.Sparse],
                         unit=sc.units.counts)
    if not drop_pulse_times:
        labs = sc.Variable([sc.Dim.Position, dim],
                           shape=[nHist, sc.Dimensions.Sparse])
    for i in range(nHist):
        coords[sc.Dim.Position, i].values = ws.getSpectrum(i).getTofs()
        if not drop_pulse_times:
            # Pulse times have a Mantid-specific format so the conversion is
            # very slow.
            # TODO: Find a more efficient way to do this.
            pt = ws.getSpectrum(i).getPulseTimes()
            labs[sc.Dim.Position, i].values = np.asarray([p.total_nanoseconds()
                                                          for p in pt])

    coords_and_labs = {"coords": {dim: coords, sc.Dim.Position: pos},
                       "labels": {"spectrum_number": num,
                                  "component_info": comp_info}}
    if not drop_pulse_times:
        coords_and_labs["labels"]["pulse_times"] = labs
    return sc.DataArray(**coords_and_labs)


def load(filename="", drop_pulse_times=False, instrument_filename=None,
         **kwargs):
    """
    Wrapper function to provide a load method for a Nexus file, hiding mantid
    specific code from the scipp interface.
    """
    import mantid.simpleapi as mantid
    ws = mantid.LoadEventNexus(filename, **kwargs)
    if instrument_filename is not None:
        mantid.LoadInstrument(ws, FileName=instrument_filename,
                              RewriteSpectraMap=True)
    if ws.id() == 'Workspace2D':
        return ConvertWorkspace2DToDataset(ws)
    if ws.id() == 'EventWorkspace':
        return ConvertEventWorkspaceToDataset(ws, drop_pulse_times)
    raise RuntimeError('Unsupported workspace type')
