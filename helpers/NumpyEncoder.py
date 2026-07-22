import json
from numpy import integer, floating, ndarray

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalar/array types to native Python types.

    Pass as ``cls=NumpyEncoder`` to ``json.dumps``/``json.dump`` so numpy
    ints, floats, and arrays (which are not JSON-serializable by default)
    are coerced automatically.
    """
    def default(self, obj):
        """Convert numpy types to JSON-serializable equivalents.

        Args:
            obj: Object that the default JSONEncoder failed to serialize.

        Returns:
            A JSON-serializable representation of ``obj``.
        """
        if isinstance(obj, integer):
            return int(obj)
        elif isinstance(obj, floating):
            return float(obj)
        elif isinstance(obj, ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)