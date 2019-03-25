import numpy as np

# MACS format pattern representing the beginning of SNP information line
START_DATA_LINE = 'SITE'
# MACS format pattern representing the end of the SNP information content in the file
LAST_LINE = 'TOTAL_SAMPLES'

class ParrallelMacsBufferReader:
    '''
    Parallel Reader for a macs format file containing information on SNP loci for multiple samples.
    Reading is done in blocks to limit memory consumtion: each time a limited range of SNP loci is read.
    Reading is also done to a limited range of samples so parallel processing of the file is possible.
    '''
    def __init__(self, filename, block_len, start_index=0, end_index=float("inf")):
        '''
            Construct a macs reader.

            Args:
                filename: file name as a string.
                block_len: number of SNP loci to read each time the reader is queried.
                start_index: the index of he first sample to process.
                end_index: the index of he last sample to process.
        '''
        self.filename = filename
        self.block_len = block_len
        self.start_index = start_index
        self.end_index = end_index
        if not self.end_index < float("inf"):
            self.end_index = self.get_data_shape()[1]


    def get_data_shape(self):
        '''
        Get total number of samples and number of SNP loci in the file.

        Returns:
           sample_num: number of samples.
           snp_num: number of SNP loci.
        '''
        sample_num, snp_num = -1, -1
        data = open(self.filename)
        for line in data:
            # Ignore irrelevant lines
            if not line.startswith(START_DATA_LINE):
                continue
            # SNP information is exhausted
            elif line.startswith(LAST_LINE):
                break
            # Count SNP information lines
            else:
                 snp_num += 1
                 if sample_num < 0:
                     sample_num = len(line.strip().split()[4])
        data.close()
        return sample_num, snp_num


    def get_snp_data(self):
        '''
            Retrieve next buffer of SNP loci information on the relevant range of sample indices.

            Returns:
                snp_data: current SNP information represented as nunpy array of shape [BUFFER_LEN, #SNPs]
        '''
        # Open MACS file 
        data = open(self.filename)
        snp_data = list()
        snp_counter = 0
        # Process next buffer of SNP information lines
        for line in data:
            # Ignore irrelevant lines
            if not line.startswith(START_DATA_LINE):
                continue
            # SNP information is exhausted
            elif line.startswith(LAST_LINE):
                break
            # Process SNP information line and store relevant data as a numpy array
            else:
                snp_info = [bit for bit in line.strip().split()[4][self.start_index:self.end_index]]
                snp_data.append(snp_info)
                snp_counter += 1
            # Block is fully read
            if snp_counter == self.block_len:
                snp_data = np.array(snp_data).astype("uint8").T
                yield snp_data
                snp_data = list()
                snp_counter = 0
        data.close()
        # Retrieve last block
        if len(snp_data) > 0:
            snp_data = np.array(snp_data).astype("uint8").T
            yield snp_data

if __name__ == "__main__":
    import time

    start = time.time()

    if True:
        K = 50
        filename = "/home/gollum/ilanben/haplotype/4k_1e7_e0.001"
    if False:
        K = 10
        filename = "input.txt"
    pmbr = ParrallelMacsBufferReader(filename, K, 0, 500)

    sample_num, snp_num = pmbr.get_data_shape()
    print sample_num, snp_num

    snp_blocks = pmbr.get_snp_data()
    block_index = 0
    for snp_block in snp_blocks:
        print block_index, snp_block.shape
        block_index += 1

    end = time.time()
    print end-start
