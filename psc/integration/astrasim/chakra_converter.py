import pandas as pd
import os
import argparse
import sys

from chakra.src.third_party.utils.protolib import encodeMessage
from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, Node, MEM_LOAD_NODE, MEM_STORE_NODE

def convert_to_chakra(parquet_path, output_path, limit=None):
    if not os.path.exists(parquet_path):
        print(f"Error: {parquet_path} not found.")
        return

    print(f"Converting {parquet_path} to Chakra ET (limit={limit})...")
    df = pd.read_parquet(parquet_path)
    if limit:
        df = df.head(limit)
    
    # Minimal Chakra ET structure
    et = GlobalMetadata()
    attr = et.attr.add()
    attr.name = "version"
    attr.int64_val = 0
    
    nodes = []
    for i, row in df.iterrows():
        node = Node()
        node.id = i
        node.name = f"mem_access_{i}"
        
        # Map operations to Chakra node types
        if row['op'] == 'read':
            node.type = MEM_LOAD_NODE
        else:
            node.type = MEM_STORE_NODE
            
        # tensor_size in bytes as an attribute
        s_attr = node.attr.add()
        s_attr.name = "tensor_size"
        s_attr.int64_val = int(row['size_kb'] * 1024)
        
        # Add a custom attribute for our phase-conditioned policy
        p_attr = node.attr.add()
        p_attr.name = "phase"
        p_attr.int64_val = int(row['phase'])
        
        b_attr = node.attr.add()
        b_attr.name = "block_id"
        b_attr.int64_val = int(row['block_id'])
        
        nodes.append(node)
        
    with open(output_path, "wb") as f:
        # First write metadata
        encodeMessage(f, et)
        # Then write nodes
        for node in nodes:
            encodeMessage(f, node)
            
    print(f"Successfully wrote {len(nodes)} nodes to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSC to Chakra ET Converter")
    parser.add_argument("--input", type=str, required=True, help="Path to input Parquet trace")
    parser.add_argument("--output", type=str, required=True, help="Path to output Chakra ET file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of nodes in the output trace")
    args = parser.parse_args()
    
    if 'GlobalMetadata' not in globals():
        print("Required chakra modules missing. Please install chakra first.")
        sys.exit(1)
        
    convert_to_chakra(args.input, args.output, args.limit)
