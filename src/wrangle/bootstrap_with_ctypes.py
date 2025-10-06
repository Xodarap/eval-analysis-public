import numpy as np
from ctypes import CDLL, c_int64, c_int32, POINTER
from pathlib import Path

# Load the compiled C library
lib_path = Path(__file__).parent / 'bootstrap_resample.so'
lib = CDLL(str(lib_path))

# Define the function signatures

# calculate_output_size(value_indices, n_indices, sampled_values, n_sampled) -> int64
lib.calculate_output_size.argtypes = [
    POINTER(c_int32),  # value_indices
    c_int64,           # n_indices
    POINTER(c_int64),  # sampled_values
    c_int64            # n_sampled
]
lib.calculate_output_size.restype = c_int64

# resample_hierarchical(...) -> int64
lib.resample_hierarchical.argtypes = [
    POINTER(c_int64),  # group_indices
    c_int64,           # n_indices
    POINTER(c_int32),  # value_indices
    POINTER(c_int64),  # sampled_values
    c_int64,           # n_sampled
    c_int32,           # new_split_id
    c_int32,           # is_last_category
    POINTER(c_int64),  # output_indices
    POINTER(c_int32)   # output_split_ids (can be None)
]
lib.resample_hierarchical.restype = c_int64


def resample_hierarchical_fast(
    group_indices: np.ndarray,
    value_indices: np.ndarray,
    sampled_values: np.ndarray,
    is_last_category: bool,
    new_split_id: int
):
    """
    Python wrapper for the C resample function.
    
    Args:
        group_indices: int64 array
        value_indices: int32 array
        sampled_values: int64 array
        is_last_category: boolean
        new_split_id: int
    
    Returns:
        tuple of (new_indices, new_split_ids) or just new_indices if last category
    """
    # Ensure arrays are contiguous and correct dtype
    group_indices = np.ascontiguousarray(group_indices, dtype=np.int64)
    value_indices = np.ascontiguousarray(value_indices, dtype=np.int32)
    sampled_values = np.ascontiguousarray(sampled_values, dtype=np.int64)
    
    n_indices = len(group_indices)
    n_sampled = len(sampled_values)
    
    # First, calculate how big the output will be
    output_size = lib.calculate_output_size(
        value_indices.ctypes.data_as(POINTER(c_int32)),
        c_int64(n_indices),
        sampled_values.ctypes.data_as(POINTER(c_int64)),
        c_int64(n_sampled)
    )
    
    if output_size < 0:
        raise MemoryError("Failed to allocate memory in C code")
    
    # Pre-allocate output arrays
    output_indices = np.empty(output_size, dtype=np.int64)
    output_split_ids = None if is_last_category else np.empty(output_size, dtype=np.int32)
    
    # Call the C function
    result_size = lib.resample_hierarchical(
        group_indices.ctypes.data_as(POINTER(c_int64)),
        c_int64(n_indices),
        value_indices.ctypes.data_as(POINTER(c_int32)),
        sampled_values.ctypes.data_as(POINTER(c_int64)),
        c_int64(n_sampled),
        c_int32(new_split_id),
        c_int32(1 if is_last_category else 0),
        output_indices.ctypes.data_as(POINTER(c_int64)),
        output_split_ids.ctypes.data_as(POINTER(c_int32)) if output_split_ids is not None else None
    )
    
    if result_size < 0:
        raise MemoryError("Failed to allocate memory in C code")
    
    if is_last_category:
        return output_indices
    else:
        return output_indices, output_split_ids


# Example usage:
if __name__ == "__main__":
    # Create test data
    group_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
    value_indices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32)
    sampled_values = np.array([0, 2, 2], dtype=np.int64)  # Sample values 0, 2, 2
    
    print("Input:")
    print(f"  group_indices: {group_indices}")
    print(f"  value_indices: {value_indices}")
    print(f"  sampled_values: {sampled_values}")
    
    # Test last category (no split IDs)
    result = resample_hierarchical_fast(
        group_indices,
        value_indices,
        sampled_values,
        is_last_category=True,
        new_split_id=0
    )
    print(f"\nOutput (last category):")
    print(f"  new_indices: {result}")
    
    # Test non-last category (with split IDs)
    result_indices, result_splits = resample_hierarchical_fast(
        group_indices,
        value_indices,
        sampled_values,
        is_last_category=False,
        new_split_id=100
    )
    print(f"\nOutput (not last category):")
    print(f"  new_indices: {result_indices}")
    print(f"  new_split_ids: {result_splits}")