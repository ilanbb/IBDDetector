import glob
import numpy as np
import random
import pickle
from tester_reader import *

def split_range(m, k):
    '''
    Split a range of m numbers (0 to m-1) into k buckets as evenly as possible

    Args:
        m: range length
        k: number of buckets

    Returns:
        buckets: list of buckets each represented a tuple of (start, end)
    '''
    lengths = [m/k for i in range(k)]
    for i in range(m % k):
        lengths[i] += 1
    buckets = list()
    curr_start = 0
    curr_end = 0
    for i in range(k):
        curr_end = curr_start + lengths[i]
        buckets.append((curr_start, curr_end))
        curr_start = curr_end
    return buckets


def gen_perm(n):
    '''
    Generate a permutation of the numbers 0, 1, 2, ..., n-1.

    Args:
        n: length of permuation.
    Returns:
        perm: list of numbers which is a valid permutaion of all inegers from 0 up to n-1.
    '''
    perm = [i for i in range(n)]
    for index in range(n-1):
        other_index = random.randint(index, n-1)
        temp = perm[other_index]
        perm[other_index] = perm[index]
        perm[index] = temp
    return perm


def gen_mask(n, k):
    '''
    Generate one mask of n bits with k bits set to one.

    Args:
        n: length of mask
        k: number of bits set to one in a mask

    Returns:
        mask: array bits representing (n,k) a mask
    '''
    perm = gen_perm(n)
    mask = [0] * n
    for position in range(k):
        mask[perm[position]] = 1
    return mask


def gen_masks(n, k, l):
    '''
    Generate a set of l masks of n bits with k bits set to one in each mask.

    Args:
        n: length of mask
        k: number of bits set to one in each mask
        l: number of masks

    Returns:
        masks: numpy array of l (n,k) masks
    '''
    masks_arr = list()
    for mask_index in range(l):
        masks_arr.append(gen_mask(n, k))
    masks = np.array(masks_arr)
    return masks


def apply_masks(data, masks, bit_converter, M, T, K, L):
    '''
    Apply L masks of T-out-of-K on a data matrix of shape [M,K]

    args:
        data: nunmpy array of data with M rows and up to K columns
        masks: numpy array of masks of shape [L,K]
        bit_converter: numpy array with first T powers of 2 (in a reverse order)
        M: number of rows in data matrix
        T: number of bit in each maks that are set
        K: mask length
        L: number of masks

    returns:
        result: result of applying L masks on each row as a numpy array of shape [M, 1, L]
                the result of applying a mask on K bits is the decimal value of the T bits sampled.
    '''
    # Pad data matrix into a block of size K in case of a prtial block
    diff = K - data.shape[1]
    if diff == 0:
        paded_data = data
    else:
        paded_data = np.concatenate((data, np.zeros((M,diff), dtype=data.dtype)), axis=1)
    # Convert data into a nunpy array of [M, L, K]
    expanded_data = np.broadcast_to(np.expand_dims(paded_data, 1), (M, L, K))
    # Apply masks and reshape to [M*L, T]
    masked_data = expanded_data[:, masks == 1]
    masked_data = masked_data.reshape((M*L,T)) 
    # Convert bit arrays into decimal numbers 
    int_masked_data = masked_data.dot(bit_converter)
    # Reshape results into a matrix of shape [M,1,L]
    result = int_masked_data.reshape((M,1,L))
    return result


def squeeze_locations(locations, max_gap):
    '''
    Given a list of location indices, shorten the list into long intervals of consecutive indices (with short gaps)

    Args:
        locations: list of location indices
        max_gap: maximum gap allowed in interval

    Returns:
        intervals: list of intervals
    '''
    locations.sort()
    intervals = list()
    interval = [locations[0], locations[0]]
    for location in locations[1:]:
        if location <= interval[1] + max_gap:
            interval[1] = location
        else:
            intervals.append((interval[0], interval[1]))
            interval = [location, location]
    intervals.append((interval[0], interval[1]))
    return intervals


def count_elements(mapping):
    '''
    Count total number of intervals in a mapping of keys to lists of elements

    Args:
        mapping: a dictionary mapping keyts to lists of elements.

    Returns:
        elems: total number of elements in the list values of the mapping.
    '''
    elems = 0
    for key in mapping:
        elems += len(mapping[key])
    return elems


def log_results(ibd_data, gap, min_match, log_file):
    '''
    Log results of IBD algorithm into a file with similarity information:
    Each line in the file is of the followinf format:
    SAMPLE1_INDEX	SAMPLE2_INDEX	START_INDEX	END_INDEX

    Args:
        ibd_data: dictionary mapping pairs of sample indices to an interval defined by a tuple od start and end indices.
        gap: the length of a skip between consecutive segments.
        min_match: the minimal length of matching segments.
        log_file: log file name.
    '''
    sorted_pairs = sorted(ibd_data.keys())
    log = open(log_file, "w")
    for pair in sorted_pairs:
        pair_str = str(pair[0]) + "\t" + str(pair[1]) + "\t"
        for interval in ibd_data[pair]:
            start_seg = gap * interval[0]
            end_seg = gap * interval[1] + min_match
            log.write(pair_str + str(start_seg) + "\t" + str(end_seg) + "\n")
    log.close()


def test_ibd_results(log_file, data_file):
    '''
    Verify correctness of IBD algorithm

    Args:
        log_file: log file name where IBD results where written to.
        data_file: file with raw data in a CVS format.
    '''
    tr = TesterReader(data_file)
    suc = 0
    total = 0
    matches = open(log_file)
    for line in matches:
        sample1_index, sample2_index, start_seg, end_seg = line.strip().split()
        sample1_index, sample2_index, start_seg, end_seg = int(sample1_index), int(sample2_index), int(start_seg), int(end_seg)
        sample1 = tr.get_line(sample1_index).strip().split(",")
        sample2 = tr.get_line(sample2_index).strip().split(",")
        seg1 = sample1[start_seg:end_seg]
        seg2 = sample2[start_seg:end_seg]
        ham_dist = sum([seg1[i]!=seg2[i] for i in range(len(seg1))])
        eps = (100.0 * ham_dist) / (end_seg-start_seg)
        print "(", sample1_index, ",", sample2_index, ")", start_seg, "-", end_seg, ham_dist, eps
        total += 1
        if eps < 2:
            suc +=1
        print "Suc:", suc, "out of", total, (100.0*suc)/total
    matches.close()
    


if __name__ == "__main__":

    data = np.array([[1,2,3,4,5], [6,7,8,9,10]])
    data = np.array([[1,2,3,4], [6,7,8,9]])
    masks = np.array([[1,1,1,1,0], [0,1,1,1,1], [1,1,1,0,1]])
    print data, "\n"
    print masks, "\n"
    bit_converter = 2**np.arange(4)[::-1]
    print apply_masks(data, masks, bit_converter, 2, 5, 4, 3)
