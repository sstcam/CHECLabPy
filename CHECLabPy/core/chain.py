import yaml
from CHECLabPy.core import child_subclasses
from CHECLabPy.core.reducer import WaveformReducer
import CHECLabPy.waveform_reducers  # Required to add reducers to global


class WaveformReducerChain:
    # Default setting for each column
    default_columns = dict(
        # Baseline
        baseline_end_mean=True,
        baseline_end_rms=True,
        baseline_start_mean=True,
        baseline_start_rms=True,
        waveform_mean=True,
        waveform_rms=True,

        # Timing
        t_event=False,
        t_pulse=False,
        t_pulse_amp=False,
        t_rise=False,
        fwhm=False,

        # Saturation
        saturation_sum=True,

        # AverageWF
        t_averagewf=True,
        charge_averagewf=True,

        # CrossCorrelation
        t_cc=True,
        charge_cc=True,
        height_cc=True,

        # CrossCorrelationLocal
        t_cc_local=False,
        charge_cc_local=False,
        height_cc_local=False,

        # CrossCorrelationNeighbour
        t_cc_nn=False,
        charge_cc_nn=False,
        height_cc_nn=False,

        # CtapipeLocalPeakIntegrator
        t_local=False,
        charge_local=False,

        # CtapipeNeighbourPeakIntegrator
        t_nn=False,
        charge_nn=False,

        # NNLSPulseExtraction
        charge_nnls=False,
        nnls_tcharge=False,
        nnls_tmcharge=False,
        nnls_tccharge=False,
        nnls_norm=False,
        nnls_npulses=False,
        nnls_errata=False,

        # SlidingWindow
        t_sliding=False,
        charge_sliding=False,

        # SlidingWindowLocal
        t_sliding_local=False,
        charge_sliding_local=False,

        # SlidingWindowNeighbour
        t_sliding_nn=False,
        charge_sliding_nn=False,

        # SPAmplitude
        sp_argmax=False,
        sp_max=False,
    )
    # Default setting for additional configuration parameters
    default_config = dict(

    )

    def __init__(self, n_pixels, n_samples, config_path=None, **kwargs):
        """
        Builds a chain of `WaveformReducers` for waveform processing to
        generate DL1 files.

        A `WaveformReducer` is only included in the chain if atleast one of its
        columns is active.

        Parameters
        ----------
        n_pixels : int
            Number of pixels in the data to be processed by the
            `WaveformReducer`.
        n_samples : int
            Number of samples in the data to be processed by the
            `WaveformReducer`.
        config_path : str
            Path to a YAML file to be used for configuration of the
            `WaveformReducer` and its columns.
        kwargs
            Columns can be deactivated by passing their "name"=False via the
            kwargs. Configuration to the `WaveformReducer` can also be
            passed via kwargs.
        """
        config = {**self.default_config}
        if config_path:
            loaded_config = self._load_config(config_path)
            config = {**config, **loaded_config, **kwargs}
        else:
            config = {**config, **self.default_columns, **kwargs}

        self.chain = self._build_chain(n_pixels, n_samples, **config)

        self.config = config

    @staticmethod
    def _load_config(path):
        """
        Read the YAML config file to configure the `WaveformReducer` and its
        columns

        Parameters
        ----------
        path : str
            Path to the YAML file.

        Returns
        -------
        config : dict
            Dictionary containing the configuration read from the YAML file.
        """
        print("Loading WaveformReducer configuration from: {}".format(path))
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
            if config is None:
                return {}
            return config

    @staticmethod
    def _build_chain(n_pixels, n_samples, **config):
        """
        Build the chain of initialised `WaveformReducers`, ready for
        waveform processing.

        Parameters
        ----------
        n_pixels : int
            Number of pixels in the data to be processed by the
            `WaveformReducer`.
        n_samples : int
            Number of samples in the data to be processed by the
            `WaveformReducer`.
        config
            Configuration to be passed to the `WaveformReducer`.

        Returns
        -------
        chain : list
            List of the `process` methods of each `WaveformReducer`.
        """
        print("Building WaveformReducerChain:")
        all_reducers = child_subclasses(WaveformReducer)
        reducer_config = {**config, '_disable_by_default': True}

        chain = []
        for r in all_reducers:
            if len(r.get_active_columns(**reducer_config)) > 0:
                reducer = r(n_pixels, n_samples, **reducer_config)
                chain.append(reducer.process)
                for c in reducer.active_columns:
                    print("\t{}.{}".format(reducer.__class__.__name__, c))
        return chain

    def process(self, waveforms):
        """
        Iterate through the chain, calling the `process` method of each
        `WaveformReducer` in the chain. Combines all the returned column dicts
        into a single dict.

        Parameters
        ----------
        waveforms : ndarray
            The waveforms to be processed.

        Returns
        -------
        dict
            Dictionary containing the return values of the columns, with a key
            corresponding to the name of the column.
        """
        return {k: v for f in self.chain for k, v in f(waveforms).items()}
