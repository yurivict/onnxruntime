import argparse
import onnx
from onnx import numpy_helper, helper
import numpy as np
import os

def create_matmul_integer(b, m, n, k, out_dir):
    input_data = np.random.randint(low=0, high=127, size=(b,m,k), dtype=np.uint8)
    weight_data = np.random.randint(low=-128, high=127, size=(k,n), dtype=np.int8)
    output_data = np.matmul(input_data.astype(np.int32), weight_data.astype(np.int32))
    mp = onnx.ModelProto()
    opset = mp.opset_import.add()
    opset.version = 10
    opset.domain = "ai.onnx"
    graph = mp.graph
    graph.input.add().CopyFrom(helper.make_tensor_value_info('Input', onnx.TensorProto.UINT8, (b,m,k)))
    graph.output.add().CopyFrom(helper.make_tensor_value_info('Output', onnx.TensorProto.INT32, (b,m,n)))
    graph.initializer.add().CopyFrom(numpy_helper.from_array(weight_data, 'W'))
    graph.node.add().CopyFrom(helper.make_node('MatMulInteger', ['Input', 'W'], ['Output']))
    
    os.makedirs(os.path.join(out_dir, 'test_data_set_0'), exist_ok=True)
    onnx.save(mp, os.path.join(out_dir, 'model.onnx'))
    onnx.save_tensor(numpy_helper.from_array(input_data, 'Input'), os.path.join(out_dir, 'test_data_set_0', 'input_0.pb'))
    onnx.save_tensor(numpy_helper.from_array(output_data, 'Output'), os.path.join(out_dir, 'test_data_set_0', 'output_0.pb'))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', required=True, help='The output directory')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--m', type=int, default=1, help='M size')
    parser.add_argument('--n', type=int, default=4096, help='N size')
    parser.add_argument('--k', type=int, default=1024, help='K size')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    create_matmul_integer(args.batch, args.m, args.n, args.k, args.outdir)
    print('Done!')