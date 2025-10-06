#include <stdint.h>
#include <stdlib.h>

/**
 * Resample indices hierarchically for bootstrap.
 * 
 * @param group_indices: Input array of indices for this group
 * @param n_indices: Length of group_indices
 * @param value_indices: Which value each index belongs to (0, 1, 2, ...)
 * @param sampled_values: Which values were sampled with replacement
 * @param n_sampled: Length of sampled_values
 * @param new_split_id: Starting ID for new splits
 * @param is_last_category: 1 if last category, 0 otherwise
 * @param output_indices: Pre-allocated output array for new indices
 * @param output_split_ids: Pre-allocated output array for split IDs (can be NULL if last category)
 * @return: Number of output indices written
 */
int64_t resample_hierarchical(
    const int64_t* group_indices,
    int64_t n_indices,
    const int32_t* value_indices,
    const int64_t* sampled_values,
    int64_t n_sampled,
    int32_t new_split_id,
    int32_t is_last_category,
    int64_t* output_indices,
    int32_t* output_split_ids
) {
    // First pass: find max value to size the count array
    int32_t max_value = 0;
    for (int64_t i = 0; i < n_indices; i++) {
        if (value_indices[i] > max_value) {
            max_value = value_indices[i];
        }
    }
    int32_t n_unique_values = max_value + 1;
    
    // Count occurrences of each value
    int64_t* count_per_value = (int64_t*)calloc(n_unique_values, sizeof(int64_t));
    if (count_per_value == NULL) {
        return -1; // Memory allocation failed
    }
    
    for (int64_t i = 0; i < n_indices; i++) {
        count_per_value[value_indices[i]]++;
    }
    
    // Second pass: fill output arrays
    int64_t output_idx = 0;
    
    for (int64_t j = 0; j < n_sampled; j++) {
        int64_t sampled_val = sampled_values[j];
        
        // Copy all indices where value_indices == sampled_val
        for (int64_t i = 0; i < n_indices; i++) {
            if (value_indices[i] == sampled_val) {
                output_indices[output_idx] = group_indices[i];
                
                if (!is_last_category) {
                    output_split_ids[output_idx] = new_split_id + j;
                }
                
                output_idx++;
            }
        }
    }
    
    free(count_per_value);
    return output_idx;
}

/**
 * Calculate the output size without actually doing the resampling.
 * Useful for pre-allocating arrays in Python.
 */
int64_t calculate_output_size(
    const int32_t* value_indices,
    int64_t n_indices,
    const int64_t* sampled_values,
    int64_t n_sampled
) {
    // Find max value
    int32_t max_value = 0;
    for (int64_t i = 0; i < n_indices; i++) {
        if (value_indices[i] > max_value) {
            max_value = value_indices[i];
        }
    }
    int32_t n_unique_values = max_value + 1;
    
    // Count occurrences
    int64_t* count_per_value = (int64_t*)calloc(n_unique_values, sizeof(int64_t));
    if (count_per_value == NULL) {
        return -1;
    }
    
    for (int64_t i = 0; i < n_indices; i++) {
        count_per_value[value_indices[i]]++;
    }
    
    // Calculate total output size
    int64_t total_output = 0;
    for (int64_t j = 0; j < n_sampled; j++) {
        total_output += count_per_value[sampled_values[j]];
    }
    
    free(count_per_value);
    return total_output;
}