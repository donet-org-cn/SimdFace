"""Prepare the pinned ONNX models for SimdFace's managed inference runtime.

Run with a directory containing the original exports and an output directory.
Requires onnx, onnxruntime and numpy for this offline conversion.
"""

import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


FAS_HASH = "b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907"
MESH_HASH = "3ca77cf59c18e4da0eccb46695bf604683fa564253e3385892981a5c274fb10f"


def check_hash(path, digest):
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Unexpected source weights: {path}")


def prepare_fas(path):
    model = onnx.load(path)
    graph = model.graph
    weights = {item.name: item for item in graph.initializer}
    for i, node in enumerate(graph.node):
        if node.op_type != "MatMul":
            continue
        matrix = numpy_helper.to_array(weights[node.input[1]]).T.copy()
        bias = np.zeros(matrix.shape[0], dtype=np.float32)
        matrix_name = f"simdface.matrix.{i}"
        bias_name = f"simdface.bias.{i}"
        graph.initializer.extend([
            numpy_helper.from_array(matrix, matrix_name),
            numpy_helper.from_array(bias, bias_name),
        ])
        graph.node[i].CopyFrom(helper.make_node(
            "Gemm", [node.input[0], matrix_name, bias_name], list(node.output), transB=1))
    names = {name for node in graph.node for name in node.input}
    retained = [item for item in graph.initializer if item.name in names]
    del graph.initializer[:]
    graph.initializer.extend(retained)
    return model


def prepare_mesh(path):
    with TemporaryDirectory() as tmp:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.optimized_model_filepath = str(Path(tmp) / "folded.onnx")
        ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        model = onnx.load(options.optimized_model_filepath)
    graph = model.graph
    weights = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
    for node in graph.node:
        if node.op_type == "Pad":
            pads = weights[node.input[1]]
            value = weights.get(node.input[2], 0) if len(node.input) > 2 else 0
            if pads.shape != (8,) or pads[0] != 0 or pads[4] != 0 or min(pads) < 0 or value != 0:
                raise ValueError("Expected constant zero padding of a single NCHW sample")
    return model


def save_verified(source, model, destination, size, max_value):
    onnx.checker.check_model(model)
    onnx.save(model, destination)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    before, after = (
        ort.InferenceSession(str(p), options, providers=["CPUExecutionProvider"])
        for p in (source, destination)
    )
    data = np.random.default_rng(44).uniform(0, max_value, (1, 3, size, size)).astype(np.float32)
    for a, b in zip(before.run(None, {"input": data}), after.run(None, {"input": data})):
        np.testing.assert_allclose(a, b, atol=0.001, rtol=0.0001)
    print(f"{destination.name} {hashlib.sha256(destination.read_bytes()).hexdigest()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fas = args.source / "MiniFASNetV2.onnx"
    mesh = args.source / "face_mesh_Nx3x192x192.onnx"
    check_hash(fas, FAS_HASH)
    check_hash(mesh, MESH_HASH)
    save_verified(fas, prepare_fas(fas), args.output / "minifasnet_v2_simdface.onnx", 80, 255)
    save_verified(mesh, prepare_mesh(mesh), args.output / "face_mesh_468_simdface.onnx", 192, 1)


if __name__ == "__main__":
    main()
