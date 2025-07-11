"""
Retry Failed Samples Processing Script - VIETNAMESE VERSION

This script processes failed samples from a JSON error log file generated by multi_vi.py.
It has the same parameter structure as multi_vi.py with the addition of failed_json_path parameter.

Key features:
- Load failed sample IDs from JSON error log
- Find and retry processing of failed samples
- Each different JSON path creates a unique split name
- Same threading and batch processing as multi_vi.py
- Maintains compatibility with original parameter structure

Usage:
    create_retry_dataset(
        failed_json_path="path/to/failed_ids.json",
        new_dataset_name="dataset_name",
        num_threads=1,
        samples_per_thread=25000,
        wait_seconds=0,
        batch_size=9,
        expected_parts=1,
        api_key_start_index=1
    )
"""

import json
import re
import os
import time
import threading
from datasets import load_dataset, Dataset
from huggingface_hub import login
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Import functions from the main file
from multi_vi import (
    single_prompt_template,
    create_batch_prompt_template,
    extract_json_from_completion,
    count_parts_in_json,
    extract_tags_from_response,
    extract_samples_from_batch_response,
    process_batch_with_gemini
)

# Retry-specific error logging for failed samples during retry processing
retry_error_log_lock = threading.Lock()
retry_failed_sample_ids = []

def log_retry_failed_sample_ids(sample_ids):
    """Log failed sample IDs in batch during retry"""
    global retry_failed_sample_ids
    with retry_error_log_lock:
        retry_failed_sample_ids.extend(sample_ids)

def save_retry_error_log(original_json_path, retry_dataset_name, retry_split_name):
    """Save retry error log with failed sample IDs in retry folder"""
    global retry_failed_sample_ids
    with retry_error_log_lock:
        # Create retry folder if it doesn't exist
        retry_folder = "retry"
        if not os.path.exists(retry_folder):
            os.makedirs(retry_folder)
            print(f"Created retry folder: {retry_folder}")
        
        if retry_failed_sample_ids:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename based on original JSON and retry info
            original_json_name = os.path.splitext(os.path.basename(original_json_path))[0]
            filename = f"{retry_folder}/retry_failed_{original_json_name}_{timestamp}.json"
            
            error_data = {
                "original_failed_json": original_json_path,
                "retry_dataset": retry_dataset_name,
                "retry_split": retry_split_name,
                "retry_failed_count": len(retry_failed_sample_ids),
                "retry_failed_sample_ids": retry_failed_sample_ids,
                "retry_timestamp": timestamp,
                "note": "These samples failed during retry processing"
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False)
            
            print(f"Retry failed IDs saved: {filename}")
            print(f"Total retry failed samples: {len(retry_failed_sample_ids)}")
        else:
            print("No retry failed samples to log!")

def load_failed_samples_from_json(json_file_path):
    """Load failed sample information from JSON file"""
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"Failed samples JSON file not found: {json_file_path}")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        error_data = json.load(f)
    
    print(f"Loaded error log from: {json_file_path}")
    print(f"Original dataset: {error_data['dataset']}")
    print(f"Original split: {error_data['split']}")
    print(f"Original start index: {error_data['start_index']}")
    print(f"Failed samples count: {error_data['failed_count']}")
    print(f"Error log timestamp: {error_data['timestamp']}")
    
    return error_data

def find_samples_by_ids(dataset, sample_ids):
    """Find samples in dataset by their IDs"""
    found_samples = []
    found_indices = []
    not_found_ids = []
    
    print(f"Searching for {len(sample_ids)} failed samples in dataset...")
    
    for idx, sample in enumerate(tqdm(dataset, desc="Searching samples")):
        sample_id = sample.get('id', f'unknown_{idx}')
        if sample_id in sample_ids:
            found_samples.append(sample)
            found_indices.append(idx)
    
    # Check for IDs that weren't found
    found_ids = set(sample.get('id', f'unknown_{idx}') for idx, sample in enumerate(found_samples))
    not_found_ids = [sid for sid in sample_ids if sid not in found_ids]
    
    print(f"Found {len(found_samples)} samples out of {len(sample_ids)} failed IDs")
    if not_found_ids:
        print(f"Warning: {len(not_found_ids)} sample IDs not found in dataset")
        print(f"Not found IDs (first 10): {not_found_ids[:10]}")
    
    return found_samples, found_indices

def process_retry_thread(dataset_samples, thread_id, api_key_index, max_parts, wait_seconds, batch_size):
    """Process retry samples in a thread using batch processing"""
    
    # Load API key
    api_key = os.getenv(f'GEMINI_API_KEY_{api_key_index}')
    if not api_key:
        print(f"API key GEMINI_API_KEY_{api_key_index} not found!")
        return []
    
    processed_data = []
    thread_failed_ids = []
    
    # Process samples in batches
    dataset_list = list(dataset_samples)
    total_samples = len(dataset_list)
    
    for batch_start in tqdm(range(0, total_samples, batch_size), desc=f"Retry Thread {thread_id}"):
        batch_end = min(batch_start + batch_size, total_samples)
        samples_batch = dataset_list[batch_start:batch_end]
        
        # Measure API call timing
        api_start_time = time.time()
        
        # Process batch with Gemini
        batch_results = process_batch_with_gemini(samples_batch, api_key, max_parts, batch_size, thread_id, batch_start)
        
        api_end_time = time.time()
        api_elapsed_time = api_end_time - api_start_time
        
        batch_failed_ids = []
        
        # Process each sample in the batch
        for idx, (sample, result) in enumerate(zip(samples_batch, batch_results)):
            # Extract sample ID 
            sample_id = sample.get('id', f'unknown_{batch_start + idx}')
            
            # Count parts in original completion
            original_json = extract_json_from_completion(sample['completion'])
            original_parts_count = count_parts_in_json(original_json) if original_json else 0
            
            # Create new sample with original data
            new_sample = {
                'id': sample.get('id', f'unknown_{batch_start + idx}'),
                'original_completion': sample['completion']
            }
            
            # Track sample success/failure and count generated samples
            generated_samples_count = 0
            is_failed = False
            
            if result is None:
                is_failed = True
                new_sample['new_length'] = 0
                # Initialize all fields with empty strings for failed samples
                for i in range(1, max_parts + 1):
                    new_sample[f'input_{i}'] = ''
                    new_sample[f'think_{i}'] = ''
                    new_sample[f'json_{i}'] = ''
            else:
                # Add new_length field
                new_sample['new_length'] = result.get('new_length', 0)
                
                # Initialize all possible fields based on max_parts and count generated samples
                for i in range(1, max_parts + 1):
                    input_val = result.get(f'input_{i}', '')
                    think_val = result.get(f'think_{i}', '')
                    json_val = result.get(f'json_{i}', '')
                    
                    new_sample[f'input_{i}'] = input_val
                    new_sample[f'think_{i}'] = think_val
                    new_sample[f'json_{i}'] = json_val
                    
                    # Count if all three fields are non-empty (successful generation)
                    if input_val.strip() and think_val.strip() and json_val.strip():
                        generated_samples_count += 1
                
                # Check if generated correct number of parts or no output
                if generated_samples_count != original_parts_count or generated_samples_count == 0:
                    is_failed = True
            
            # Track failed sample ID
            if is_failed:
                batch_failed_ids.append(sample_id)
            
            processed_data.append(new_sample)
        
        # Log failed IDs for this batch
        if batch_failed_ids:
            thread_failed_ids.extend(batch_failed_ids)
        
        # Intelligent wait based on actual API time
        if batch_end < total_samples:  # Don't wait after last batch
            if api_elapsed_time < wait_seconds:
                remaining_wait = wait_seconds - api_elapsed_time
                time.sleep(remaining_wait)
    
    # Log all failed IDs for this thread using retry-specific logging
    if thread_failed_ids:
        log_retry_failed_sample_ids(thread_failed_ids)
        print(f"Retry Thread {thread_id}: {len(thread_failed_ids)} failed samples")
    
    return processed_data

def load_environment():
    """Load environment variables"""
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("Please set HF_TOKEN environment variable")
    login(token=hf_token)

def retry_failed_samples(
    failed_json_path,
    new_dataset_name,
    num_threads,
    samples_per_thread,
    wait_seconds,
    batch_size,
    expected_parts=None,
    api_key_start_index=1
):
    """
    Retry processing failed samples from JSON error log
    
    Args:
        failed_json_path: Path to the JSON file containing failed sample IDs
        new_dataset_name: Name for the new dataset to push results
        num_threads: Number of threads for parallel processing
        samples_per_thread: Maximum samples to process per thread
        wait_seconds: Minimum interval between API calls
        batch_size: Number of samples per batch/API call
        expected_parts: Expected number of parts in dataset (None if not checking)
        api_key_start_index: Starting index for API keys
    """
    
    # Load environment
    load_environment()
    
    # Load failed samples information from JSON
    error_data = load_failed_samples_from_json(failed_json_path)
    
    # Extract information from error log
    original_dataset = error_data['dataset']
    original_split = error_data['split']
    failed_sample_ids_from_json = error_data['failed_sample_ids']
    original_start_index = error_data.get('start_index', 0)
    
    if not failed_sample_ids_from_json:
        print("No failed sample IDs found in the JSON file!")
        return
    
    print(f"Retrying {len(failed_sample_ids_from_json)} failed samples...")
    
    # Load original dataset
    print(f"Loading original dataset: {original_dataset}, split: {original_split}")
    dataset = load_dataset(original_dataset, split=original_split)
    
    # Find failed samples in the dataset
    failed_samples, failed_indices = find_samples_by_ids(dataset, failed_sample_ids_from_json)
    
    if not failed_samples:
        print("No failed samples found in the dataset!")
        return
    
    # Determine max_parts
    if expected_parts:
        max_parts = expected_parts
        print(f"Using expected_parts: {max_parts}")
        
        # Validate that failed samples have expected parts
        parts_distribution = {}
        for sample in failed_samples:
            completion = sample.get('completion', '')
            json_data = extract_json_from_completion(completion)
            if json_data:
                parts_count = count_parts_in_json(json_data)
                if parts_count in parts_distribution:
                    parts_distribution[parts_count] += 1
                else:
                    parts_distribution[parts_count] = 1
        
        print(f"Parts distribution in failed samples: {parts_distribution}")
        if expected_parts not in parts_distribution:
            print(f"Warning: No failed samples have expected {expected_parts} parts!")
        
    else:
        # Survey failed samples to determine max_parts
        max_parts = 0
        for sample in failed_samples:
            completion = sample.get('completion', '')
            json_data = extract_json_from_completion(completion)
            if json_data:
                parts_count = count_parts_in_json(json_data)
                max_parts = max(max_parts, parts_count)
        
        if max_parts == 0:
            print("Cannot determine max_parts from failed samples!")
            return
        
        print(f"Auto-detected max_parts from failed samples: {max_parts}")
    
    # Calculate samples per thread similar to multi_vi.py
    total_available_samples = len(failed_samples)
    total_samples_to_process = min(total_available_samples, num_threads * samples_per_thread)
    actual_samples_per_thread = total_samples_to_process // num_threads
    
    print(f"Total failed samples available: {total_available_samples}")
    print(f"Processing {total_samples_to_process} failed samples with {num_threads} threads ({actual_samples_per_thread} samples per thread)")
    print(f"Using batch processing: {batch_size} samples per API call")
    print(f"API keys: GEMINI_API_KEY_{api_key_start_index} to GEMINI_API_KEY_{api_key_start_index + num_threads - 1}")
    
    # Split failed samples for threads
    thread_datasets = []
    for i in range(num_threads):
        thread_start_idx = i * actual_samples_per_thread
        thread_end_idx = min((i + 1) * actual_samples_per_thread, total_samples_to_process)
        
        if thread_start_idx < total_samples_to_process:
            thread_samples = failed_samples[thread_start_idx:thread_end_idx]
            thread_datasets.append(thread_samples)
        else:
            break
    
    # Reset global retry_failed_sample_ids for this retry
    global retry_failed_sample_ids
    retry_failed_sample_ids = []
    
    # Process with threads
    all_processed_data = []
    
    with ThreadPoolExecutor(max_workers=len(thread_datasets)) as executor:
        futures = []
        
        for thread_id, thread_samples in enumerate(thread_datasets):
            api_key_index = api_key_start_index + thread_id
            
            future = executor.submit(
                process_retry_thread,
                thread_samples,
                thread_id,
                api_key_index,
                max_parts,
                wait_seconds,
                batch_size
            )
            futures.append((future, thread_id, len(thread_samples)))
        
        # Collect results
        for future, thread_id, sample_count in futures:
            try:
                thread_results = future.result()
                all_processed_data.extend(thread_results)
                print(f"Retry Thread {thread_id} completed: {len(thread_results)} samples processed")
            except Exception as e:
                print(f"Retry Thread {thread_id} failed with exception: {str(e)}")

    if not all_processed_data:
        print("No data was processed successfully!")
        return

    print(f"Total processed samples: {len(all_processed_data)}")
    
    # Ensure all samples have the same structure
    if all_processed_data:
        # Create a template with all required fields
        field_template = {
            'id': '',
            'original_completion': '',
            'new_length': 0
        }
        for i in range(1, max_parts + 1):
            field_template[f'input_{i}'] = ''
            field_template[f'think_{i}'] = ''
            field_template[f'json_{i}'] = ''
        
        # Ensure all samples have all fields
        for sample in all_processed_data:
            for field_name, default_value in field_template.items():
                if field_name not in sample:
                    sample[field_name] = default_value
        
        print(f"Dataset structure validated. Each sample has {len(field_template)} fields.")
    
    # Create new dataset
    new_dataset = Dataset.from_list(all_processed_data)
    
    # Create unique split name based on JSON file path and content
    json_filename = os.path.basename(failed_json_path)
    json_name_without_ext = os.path.splitext(json_filename)[0]
    
    # Extract key info from JSON filename if it follows the pattern: failed_ids_dataset_split_startindex_timestamp.json
    if json_name_without_ext.startswith('failed_ids_'):
        # Remove 'failed_ids_' prefix
        name_parts = json_name_without_ext[11:]  # Remove 'failed_ids_'
        retry_split_name = f"retry_{name_parts}_{len(failed_sample_ids_from_json)}samples"
    else:
        # Fallback for custom JSON filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        retry_split_name = f"retry_{json_name_without_ext}_{original_split}_{len(failed_sample_ids_from_json)}samples_{timestamp}"
    
    # Save retry error log if there are failed samples (after creating split name)
    save_retry_error_log(failed_json_path, new_dataset_name, retry_split_name)
    
    print(f"Retry split name: {retry_split_name}")
    print(f"Pushing retry dataset to {new_dataset_name} with split {retry_split_name}")
    
    new_dataset.push_to_hub(new_dataset_name, split=retry_split_name)
    print("Retry dataset pushed successfully!")
    print(f"📤 Uploaded retry split: {retry_split_name}")
    
    # Calculate success rate
    original_failed_count = len(failed_sample_ids_from_json)
    new_failed_count = len(retry_failed_sample_ids)  # Global variable updated during retry processing
    success_count = original_failed_count - new_failed_count
    success_rate = (success_count / original_failed_count) * 100 if original_failed_count > 0 else 0
    
    print(f"\n📊 Retry Results Summary:")
    print(f"Original failed samples: {original_failed_count}")
    print(f"Successfully processed: {success_count}")
    print(f"Still failed: {new_failed_count}")
    print(f"Success rate: {success_rate:.1f}%")

def create_retry_dataset(failed_json_path, new_dataset_name, num_threads, samples_per_thread, wait_seconds, batch_size, expected_parts=None, api_key_start_index=1):
    """Main function to create retry dataset from failed samples JSON - matches multi_vi.py structure"""
    
    retry_failed_samples(
        failed_json_path=failed_json_path,
        new_dataset_name=new_dataset_name,
        num_threads=num_threads,
        samples_per_thread=samples_per_thread,
        wait_seconds=wait_seconds,
        batch_size=batch_size,
        expected_parts=expected_parts,
        api_key_start_index=api_key_start_index
    )

# Example usage for VIETNAMESE
if __name__ == "__main__":
    # Parameters for retry - VIETNAMESE VERSION
    failed_json_path = "failed_ids_wanhin_cad_reasoning_part_vi_part_2_0_20250704_060325.json"  # Path to your failed samples JSON
    new_dataset_name = "wanhin/retry_output_reasoning_part_2"
    num_threads = 1
    samples_per_thread = 200000  # Maximum samples to process per thread
    wait_seconds = 0  # Minimum interval between API calls
    batch_size = 4  # Number of samples per batch/API call
    expected_parts = 2  # Expected number of parts in dataset (None if not checking)
    api_key_start_index = 1  # Start from GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.
    
    create_retry_dataset(
        failed_json_path=failed_json_path,
        new_dataset_name=new_dataset_name,
        num_threads=num_threads,
        samples_per_thread=samples_per_thread,
        wait_seconds=wait_seconds,
        batch_size=batch_size,
        expected_parts=expected_parts,
        api_key_start_index=api_key_start_index
    ) 