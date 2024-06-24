# ELBA
## Parallel String Graph Construction, Transitive Reduction, and Contig Generation for De Novo Genome Assembly

## Prerequisites

1. Operating System.
  * ELBA is tested and known to work on the following operating systems.
    *  SUSE Linux Enterprise Server 15.
    *  Ubuntu 14.10.
    *  MacOS.
    
2. GCC/G++ version 8.2.0 or above.

3. Make

# Build and Run ELBA

To use ELBA on perlmutter, follow these steps:

    * Download prerequisites:

        $> git clone https://github.com/PASSIONLab/ELBA . && cd ELBA
        $> git clone https://github.com/PASSIONLab/CombBLAS
        $> git clone https://github.com/CornellHPC/HySortK.git

    * Load compiler:

        $> module load PrgEnv-gnu

    * In order to compile ELBA, you have to provide the k-mer size, lower, and upper frequency
      bounds as inputs. For example, if the k-mer size was 31, the lower frequency bound 15,
      and upper frequency bound 35, you would do the following:

        $> make K=31 L=15 U=35 -j8

    * The executable is named "elba" and is found in the ELBA directory.

    * To run ELBA on a FASTA dataset named "reads.fa", first index the file with
      the command:
        $> module load spack
        $> spack load samtools
        $> samtools faidx reads.fa

      ELBA will crash if the input FASTA is not indexed. You may need to run the following
      command prior to loading spack:
        $> module load cpu

      which will disable the gpu module. However, remember to reload the gpu module before
      running elba binary, or an error may occur during runtime.

    * ELBA must be run with a square number of processors. Suppose we want to run ELBA
      with 64 MPI tasks on a single perlmutter node. The slurm command would then be

        $> srun -N 1 -n 64 -c 2 --cpu_bind=cores ./elba reads.fa

    * Each perlmutter node has 128 cores to which we can bind each MPI tasks. Here are
      some further examples of valid slurm commands for ELBA:

        # 4 MPI tasks
        $> srun -N 1 -n 4 --cpu_bind=cores ./elba reads.fa

        # 1024 MPI tasks
        $> srun -N 8 -n 1024 -c 2 --cpu_bind=cores ./elba reads.fa

       etc.

    * ELBA can also take further input parameters as follows:

        Usage: elba [options] <reads.fa>
        Options: -x INT   x-drop alignment threshold [15]
                 -A INT   matching score [1]
                 -B INT   mismatch penalty [1]
                 -G INT   gap penalty [1]
                 -o STR   output file name prefix "elba"
                 -h       help message

## Input data samples
A few input data sets can be downloaded [here](https://portal.nersc.gov/project/m1982/dibella.2d/inputs/). If you have your own FASTQs, you can convert them into FASTAs using [seqtk](https://github.com/lh3/seqtk):

  ```
    cd ../seqtk
    ./seqtk seq -a <name>.fastq/fq > <name>.fa
  ```
A tiny example `ecsample-sub1.fa` can be found in this repository.

## Ready to run
The parameters and options of ELBA are as follows:
- ```-i <string>```: Input FASTA file.
- ```-c <integer>```: Number of sequences in the FASTA file.
- ```--sc <integer>```: Seed count. ```[default: 2]```
- ```-k <integer>```: K-mer length.
- ```-s <integer>```: K-mers stride. ```[default: 1]```
- ```--ma <integer>```: Base match score (positive). ```[default: 1]```
- ```--mi <integer>```: Base mismatch score (negative). ```[default: -1]```
- ```-g <integer>```: Gap open penalty (negative). ```[default: 0]```
- ```-e <integer>```: Gap extension penalty (negative). ```[default: -1]```
- ```-O <integer>```: Number of bytes to overlap when reading the input file in parallel. ```[default: 10000]```
- ```--afreq <integer>```: Alignment write frequency.
- ```--na```: Do not perform alignment.
- ```--fa```: Full Smith-Waterman alignment.
- ```--xa <integer>```: X-drop alignment with the indicated drop value.
- ```--of <string>```: Overlap file.
- ```--af <string>```: Output file to write alignment information. 
- ```--idxmap <string>```: Output file for input sequences to ids used in ELBA.
- ```--alph <dna|protein>```: Alphabet.

## Run test program
You can run the test dataset ```ecsample-sub1.fa``` as follows on one node (it's too small to run on multiple nodes), this command runs ELBA using x-drop alignment and ```x = 5```:
```
export OMP_NUM_THREADS=1
mpirun -np 1 ./elba -i /path/to/ecsample-sub1.fa -k 17 --idxmap elba-test -c 135 --alph dna --of overlap-test --af alignment-test -s 1 -O 100000 --afreq 100000 --xa 5
```
To run on multiple nodes, for example on 4 nodes using 4 MPI rank/node, please download ```ecsample30x.fa``` from [here](https://portal.nersc.gov/project/m1982/dibella.2d/inputs/) and run as follows:
```
export OMP_NUM_THREADS=1
mpirun -np 16 ./elba -i /path/to/ecsample30x.fa -k 17 --idxmap elba-ecsample -c 16890 --alph dna --of overlap-ecsample --af alignment-ecsample -s 1 -O 100000 --afreq 100000 --xa 5
```
You need to use a perfect square number of processes to match our 2D decomposition. Recall ```-c``` should match the number of sequences in the input FASTA.

# Citation
To cite our work or to know more about our methods, please refer to:

> Giulia Guidi, Oguz Selvitopi, Marquita Ellis, Leonid Oliker, Katherine Yelick, Aydın Buluç. [Parallel String Graph Construction and Transitive Reduction for De Novo Genome Assembly](https://arxiv.org/pdf/2010.10055.pdf). Proceedings of the IPDPS, 2021.

> Giulia Guidi, Gabriel Raulet, Daniel Rokhsar, Leonid Oliker, Katherine Yelick, Aydın Buluç. [Distributed-Memory Parallel Contig Generation for De Novo
Long-Read Genome Assembly](https://arxiv.org/pdf/2207.04350.pdf). Proceedings of the ICPP, 2022.

Further design choices and results in terms of accuracy can be found here:

> Giulia Guidi, Marquita Ellis, Daniel Rokhsar, Katherine Yelick, Aydın Buluç. [BELLA: Berkeley Efficient Long-Read to Long-Read Aligner and Overlapper](https://drive.google.com/file/d/132i0RAKyIIWk_BEl1jpf9R_V5eVkKkxT/view). bioRxiv 464420; doi: https://doi.org/10.1101/464420. Proceedings of the SIAM ACDA, 2021.

# Copyright

diBELLA 2D: Parallel String Graph Construction and Transitive Reduction for De Novo Assembly (diBELLA 2D) Copyright (c) 2021, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy).  All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE. This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights. As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.
