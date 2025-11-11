import json
import shutil
from pathlib import Path

import h5py


def _normalize_dtype(value):
    if isinstance(value, dict) and value.get("class_name") == "DTypePolicy":
        return value.get("config", {}).get("name", "float32")
    return value


def _walk(obj):
    if isinstance(obj, dict):
        for key, val in list(obj.items()):
            if key == "dtype":
                obj[key] = _normalize_dtype(val)
            else:
                obj[key] = _walk(val)
        return obj
    if isinstance(obj, list):
        return [_walk(item) for item in obj]
    return obj


def _convert_inbound_nodes(nodes):
    def add_triplet(target, history, kwargs=None):
        triplet = list(history[:3])
        if kwargs:
            triplet.append(kwargs)
        target.append(triplet)

    def process(entry, collector, inherited_kwargs=None):
        if entry is None:
            return
        if isinstance(entry, dict):
            if entry.get("class_name") == "__keras_tensor__":
                add_triplet(collector, entry["config"]["keras_history"], inherited_kwargs)
                return
            args = entry.get("args", [])
            kwargs = entry.get("kwargs", {}) or inherited_kwargs
            if isinstance(args, (list, tuple)):
                for arg in args:
                    process(arg, collector, kwargs)
            else:
                process(args, collector, kwargs)
            return
        if isinstance(entry, str):
            add_triplet(collector, [entry, 0, 0], inherited_kwargs)
            return
        if isinstance(entry, (list, tuple)):
            if entry and isinstance(entry[0], (list, tuple)):
                for sub in entry:
                    process(sub, collector, inherited_kwargs)
            elif entry and isinstance(entry[0], dict):
                for sub in entry:
                    process(sub, collector, inherited_kwargs)
            else:
                add_triplet(collector, entry, inherited_kwargs)
            return
        raise ValueError(f"Unsupported node entry type: {type(entry)}")

    converted_nodes = []
    for node in nodes:
        collector = []
        process(node, collector)
        converted_nodes.append(collector)
    return converted_nodes


def convert_model_config(h5_path: Path) -> None:
    with h5py.File(h5_path, "r") as h5_file:
        config = json.loads(h5_file.attrs["model_config"])

    config = _walk(config)

    for layer in config["config"].get("layers", []):
        inbound_nodes = layer.get("inbound_nodes", [])
        if inbound_nodes:
            layer["inbound_nodes"] = _convert_inbound_nodes(inbound_nodes)
        layer_config = layer.get("config", {})
        dtype_value = layer_config.get("dtype")
        if isinstance(dtype_value, dict) and dtype_value.get("class_name") == "DTypePolicy":
            layer_config["dtype"] = dtype_value.get("config", {}).get("name", "float32")

    inputs = config["config"].get("input_layers")
    if inputs and isinstance(inputs[0], str):
        config["config"]["input_layers"] = [inputs]

    outputs = config["config"].get("output_layers")
    if outputs and isinstance(outputs[0], str):
        config["config"]["output_layers"] = [outputs]

    new_config_str = json.dumps(config)

    with h5py.File(h5_path, "a") as h5_file:
        h5_file.attrs.modify("model_config", new_config_str)


def main():
    src = Path(
        "Diabetes-detection-using-fingerprint/Diabetes_Research-master/models/diabetes_model.h5"
    ).resolve()

    if not src.exists():
        raise FileNotFoundError(f"Model file not found at {src}")

    backup = src.with_name(src.stem + "_legacy_backup.h5")
    if not backup.exists():
        shutil.copy2(src, backup)
        print(f"Backup created at {backup}")
    else:
        print(f"Backup already exists at {backup}")

    convert_model_config(src)
    print("Model config normalized successfully.")


if __name__ == "__main__":
    main()

