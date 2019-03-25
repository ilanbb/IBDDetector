import os
import time
import psutil
import numpy as np
import numpy.random
from utils import *
from parallel_macs_reader import *
from tester_reader import *
from multiprocessing import Pool, Lock, Manager, Event


def get_blocks_lsh(params, start_index, end_index):
    '''
    Generator for a locallity sensitive hashing (LSH) computation for Hamming distance of sequences.
    A set of M sample sequences is split into blocks of length K, each is hashed with L hash functions (masks) of length T.

        Args:
            params: dictionary of all algorithm parameters
            start_index: the index of the first input sample that should be processed (inclusive)
            end_index: the index of the last input sample that should be processed (exclusive)
            worker_index: index of worker (relative to other worker of the same type)

        Returns:
            hashes: matrix of shape [M,1,L] with the hash values of the L hash functions applied on the nexr M blocks of length K.
    '''
    M = end_index - start_index
    # Get all non-overlapping sets of kmers in the data through a generator
    pmbr = ParrallelMacsBufferReader(params['data_file'], params['K'], start_index, end_index)
    blocks = pmbr.get_snp_data()
    # Apply L hash functions (masks) on each block
    for block in blocks:
        hashes = apply_masks(block, params['masks'], params['bit_converter'], M, params['T'], params['K'], params['L'])
        yield hashes


def hash_producer(all_args):
    '''
    Worker computing LSH values for estimating Hamming distance of Q-long segments for a limited range of M samples.
    The worker is provided with hash values produced by a hash value generator.

    For each seqment of length Q a sketch of L values is computed.
    Each sketch is the sum of Q/K hash values that are computed for blocks of length K.
    The summation is done in an incremental manner. Every segment has an overlapo of up to (Q/K)-1 blocks with its predecessor.

    Once the worker computes the next batch of LSH sketches for a segment of length Q for its own range of samples, 
    it waits until all worlkers are done with their samples and that all these sketches are merged and read by subsequent matcher process.

    Args:
        all_args: a list of arguments containing:
            params: dictionary of all algorithm parameters
            start_index: the index of the first input sample that should be processed (inclusive)
            end_index: the index of the last input sample that should be processed (exclusive)
            worker_index: index of worker (relative to other worker of the same type)

    Returns (via shared memory and signaling):
        lsh_sketches: the next batch of M L-long LSH values sketches computed for the next overlapping batch of data of shape [M, Q]
    '''    
    params, start_index, end_index, worker_index = all_args
    round_index = 0
    # construct a generator for hash blocks
    hash_blocks = get_blocks_lsh(params, start_index, end_index)
    # Compute number of blocks per minimal length segment
    blocks_per_segment = params['min_match_value'] / params['K']
    # Construct LSH values for the first segment as a numpy array of shape [M,Q,L]
    segment_values_lst = [None]*blocks_per_segment
    for block_index in range(blocks_per_segment):
        segment_values_lst[block_index] = hash_blocks.next()
    segment_values = np.concatenate(segment_values_lst, axis=1)
    # Obtain LSH sketch for the first minimal length segment by summing the block values (numpy array of shape [M, L])
    lsh_sketches = np.sum(segment_values, axis=1)
    # Store LSH sketches for current segment and (possibly) copy to matcher data structure
    round_index = finalize_hasher_round(worker_index, round_index, lsh_sketches)
    # Repeat computation in a sliding-window fashion with possibly skipping several blocks between two segments
    last_block_index = 0
    winnow_counter = 0
    for hash_block in hash_blocks:
        full_jump_completed = False
        # Wait until all other hash producers finished computing LSH sketches for previous segment
        if winnow_counter == 0:
            free_wait = free_producers_events[worker_index].wait(params['TIMEOUT'])
            assert(free_wait)
        winnow_counter += 1
        # Incremental update of the sum of the new block hash values 
        lsh_sketches = lsh_sketches - segment_values[:, last_block_index, :]
        segment_values[:, last_block_index, :] = np.squeeze(hash_block, axis=1)
        lsh_sketches = lsh_sketches + segment_values[:, last_block_index, :]
        # Mark the newly added matrix of sketches as the last column
        last_block_index += 1
        if last_block_index == blocks_per_segment:
           last_block_index = 0
        # Aggregate hash values until next window to report on
        if winnow_counter < params['S']:
            continue
        full_jump_completed = True
        winnow_counter = 0
        # Store LSH sketches for current segment and (possibly) copy to matcher data structure
        round_index = finalize_hasher_round(worker_index, round_index, lsh_sketches)
    # Update data structures for the case of last window reached without a full jump
    if not full_jump_completed:
        # Store LSH sketches for current segment and (possibly) copy to matcher data structure
        round_index = finalize_hasher_round(worker_index, round_index, lsh_sketches)
    # Mark end of data processing
    all_producers_done_event.set()


def finalize_hasher_round(worker_index, round_index, lsh_sketches):
    '''
    Finalize a round of LSH values computations of one worker.
    This includes writing hasher results to hashers shared memory and copying round results to a subsequent matcher.

    Args:
        worker_index: index of worker (relative to other workers of the same type).
        round_index: index of completed round of hash computation.
        lsh_sketches: the output of the current hasher worker (LSH sketches of length L for each row).

    Returns:
        The new round index,
    '''
    # Store computed sketches of current worker in shared memory of LSH sketches
    mpc_data['sketches'][worker_index] = lsh_sketches
    # Clear the event signaling that current hasher is free for processing
    free_producers_events[worker_index].clear()
    # Mark the index of the completed round for current worker
    mpc_data['producers_status'][worker_index] = round_index

    # Signal if all hash producers finished computing current batch of sketches
    all_completed_round = True
    for w_index in range(params['num_producers']):
        if mpc_data['producers_status'][w_index] != round_index:
            all_completed_round = False
            break
   
    # Aggregate all sketches in case all workers are done for current batch.
    # Information is stored as a tuple of (round_index, sketches as a numpy array of shape [#SAMPLES, #HASH_FUNCS])
    # Synchronization ensures that only one worker update shared memory for subsequent matchers. 
    # After updating the shared memory, all hash producers are marked as free for processing next batch of SNP information.
    if all_completed_round:
        producer_lock.acquire()
        if mpc_data['current_producer_round'].value == round_index:
            print "Hash producers finished round", round_index
            mpc_data['all_sketches'].append((round_index, np.concatenate(mpc_data['sketches'], axis=0)))
            mpc_data['current_producer_round'].value += 1
            for w_index in range(params['num_producers']):
                free_producers_events[w_index].set()
        producer_lock.release()
    round_index += 1
    return round_index



def ibd_worker(all_args):
    '''
    IBD detection worker which assigns relevant funtion to be executed by a process

    Args:
        all_args: function type indicator (as first argument) and the arguments required for that function

    Returns:
        The output of the relevant function
    '''
    worker_type = all_args[0]
    # Parallel LSH sketch computation (parallel row processing)
    if worker_type == 0:
        return hash_producer(all_args[1:])
    # LSH sketches analysis to fins similarities between segments of different samples
    elif worker_type == 1:
        return sketch_analyzer(all_args[1:])
    # Aggregation of consecutive similar segments between pairs of samples
    elif worker_type == 2:
        return intervals_merger(all_args[1:])


def sketch_analyzer(all_args):
    '''
    Analyze LSH sketch and store information on matching segments.
    The method pops a sketch from shared memory and writes near-duplicates information to shared memory.

    Args:
        all_args: function type indicator as first argument
    '''

    worker_index = all_args[0]
    while True:
        # Process set of LSH sketches of shape [M, L] (if exists)
        try:
            round_index, lsh_sketches = mpc_data['all_sketches'].pop(0)
        except IndexError:
            lsh_sketches = None
        if lsh_sketches is not None:
            print "Sketch analyzer", worker_index, "working on round", round_index
            round_duplicates = dict()
            # Apply OR amplification on LSH sketches
            for func_index in xrange(params['L']):
                # Construct list of duplcaites of current hash function
                worker_duplicates = dict()
                for sample_index in xrange(params['M']):
                    lsh_value = lsh_sketches[sample_index, func_index]
                    lsh_dups = worker_duplicates.get(lsh_value, list())
                    lsh_dups.append(sample_index)
                    worker_duplicates[lsh_value] = lsh_dups
                # Add near-duplicate pairs
                for lsh_value in worker_duplicates:
                    dups = worker_duplicates[lsh_value]
                    dups.sort()
                    for first_index in xrange(len(dups)-1):
                        for second_index in xrange(first_index+1, len(dups)):
                            pair = (dups[first_index], dups[second_index])
                            round_duplicates[pair] = round_duplicates.get(pair, 0) + 1
            # Filter pairs according to counts
            filtered_round_duplicates = list()
            for pair in round_duplicates:
                if round_duplicates[pair] >= params['func_threshold']:
                    filtered_round_duplicates.append(pair)

            mpc_data['duplicates'].append((round_index, filtered_round_duplicates))
            print "Sketch analyzer added round", round_index, "to duplicates data."

        # Quit if all work is done
        if len(mpc_data['all_sketches']) == 0 and all_producers_done_event.is_set():
            mpc_data['sketch_analyzers_done'][worker_index] = True
            if all(mpc_data['sketch_analyzers_done']):
                all_analyzers_done_event.set()
                break
        # Wait till next round
        time.sleep(0.01)


def intervals_merger(all_args):
    '''
    Merge near duplicate segments into maximal segmengts of near duplicates
    The method pops information on duplicates in current round from shared memory and merges them.
    Information is stored in a dictionary mapping pairs of samples to a list of maximal intervals
    '''
    round_index = 0
    # Compute the difference between two consecutive rounds in terms of number of SNPs
    gap = params['S'] * params['K']
    pairs_info = dict()
    while True:
        # Aggregate similarity information in a mapping between pair and list of round indices
        try:            
            round_index, round_pairs = mpc_data['duplicates'].pop()
            print "Merging intervals for round", round_index, len(round_pairs)
            for pair in round_pairs:
                pairs_info[pair] = pairs_info.get(pair, list())
                pairs_info[pair].append(round_index)
        except IndexError:
            pass
        # When all information is gathered - merge round indices to form maximal inervals of similarity
        if all_analyzers_done_event.is_set() and len(mpc_data['duplicates']) == 0:
            # Compute statistics on number of similar segments
            similarities_num = count_elements(pairs_info)
            print "Pairs number", len(pairs_info), "Total Similarities:", similarities_num           
            # Merge indices to maximal intervals
            for pair in pairs_info:
                pairs_info[pair] = squeeze_locations(pairs_info[pair], params['min_match_value']/gap)
            # Compute statistics on number of similar maximal intervals
            intervals_num = count_elements(pairs_info)
            print "Pairs number:", len(pairs_info), "Total Intervals:", intervals_num
            
            # Log IBD results
            print "Logging"
            log_results(pairs_info, gap, params['min_match_value'], params['log_file'])
            break

        # Wait till next round
        time.sleep(0.01)


start = time.time()

# Memory monitoring object
process = psutil.Process(os.getpid())

# Problem parameters
params = dict()
params['data_file'] = "/home/gollum/ilanben/haplotype/4k_1e7_e0.001"
params['log_file'] = "output.txt"
# Length of minimal segment 
params['min_match_value'] = 10000
# Length of block
params['K'] = 50
# Legnth of a mask
params['T'] = 3
# Number of hash functions (masks)
params['L'] = 20
params['func_threshold'] = 7#8
# Number of hash producers
params['num_producers'] = 10
# Number of sketch analyzers
params['num_analyzers'] = min(params['num_producers'], params['L'])
# Maximal timeout for synchronization between processes
params['TIMEOUT'] = 160
# Number of blocks skipped between two consecutive segments
params['S'] = 30

# Generate hash functions (as block masks)
params['masks'] = gen_masks(params['K'], params['T'], params['L'])

# Construct array of first T powers of 2 (in a reverse manner) for conversion between bit array and decimal values
params['bit_converter'] = 2**np.arange(params['T'])[::-1]

# Get numbre of samples
pmbr = ParrallelMacsBufferReader(params['data_file'], params['K'])
params['M'] = pmbr.get_data_shape()[0]


# Multiprocessing synchronization objects
producer_lock = Lock()
analyzer_lock = Lock()

# Event indicating that all hash producers are done with proessing current segment
all_producers_done_event = Event()

# Event indicating that all sketch analyzers are done
all_analyzers_done_event = Event()

# List of events indicating for every hash producer if it is idle (set) or working (clear).
free_producers_events = [Event() for i in range(params['num_producers'])]
for e in free_producers_events:
    e.set()


# Manager for constructing shared-memory objects
manager = Manager()

# Dictionary holding shared-memory objects produced by server process manager
mpc_data = dict()

# Shared list of LSH sketches for all samples (tuples of round index and sketches)
mpc_data['all_sketches'] = manager.list()

# Shared list of LSH sketches for each of the hash producers 
mpc_data['sketches'] = manager.list([None for worker_index in range(params['num_producers'])])

# Shared list with the index of the last completed round of LSH computation for each of the hash producers
mpc_data['producers_status'] = manager.list([-1 for worker_index in range(params['num_producers'])])

# Shared list of booleans indicating whether sketch analyzers are done processing LSH sketches
mpc_data['sketch_analyzers_done'] = manager.list([False for worker_index in range(params['num_analyzers'])])

# Shared counter for current producer round index 
mpc_data['current_producer_round'] = manager.Value('i', 0)

# Shared memory of near-duplicate information (tuples of round index and list of similar pairs of samples)
mpc_data['duplicates'] = manager.list()

# Construct inputs for hash producers
hasher_chunks = split_range(params['M'], params['num_producers'])
parallel_inputs = [(0, params, hasher_chunks[i][0], hasher_chunks[i][1], i) for i in range(params['num_producers'])]

# Construct inputs for sketch analyzers
parallel_inputs += [(1, i) for i in range(params['num_analyzers'])]

# Construct input for the interval merger
parallel_inputs += [(2,)]

# Launch parallel processing of all types of workers
pool=Pool(processes=params['num_producers'] + params['num_analyzers'] + 2)
results=pool.map(ibd_worker, parallel_inputs)

# Output statistics    
print "Rounds completed:", mpc_data['current_producer_round'].value
end = time.time()

print "Memory:", (process.get_memory_info()[0] / float(10**9))
print "MinHash Rolling took with MinHash", str(end-start), "seconds."

# Verify results
if True:
    test_ibd_results(params['log_file'], "/home/gollum/ilanben/haplotype/rapid1.txt")
