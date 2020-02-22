CUDA_INCLUDE_DIR=-I/usr/local/cuda/samples/common/inc -I/usr/local/cuda-10.2/include
CUDA_LIB_DIR=-L/usr/local/cuda-10.2/lib64

CUDALIBS=-lcublas -lcudart
utilS=-lpthread -lm -lblas -lhdf5_serial

CFLAGS= -O3 -I/usr/include/hdf5/serial  -I/usr/local/cuda-10.2/include -Wno-sign-compare -march=native  -Wno-format  -DUSE_SSE2  -D__USE_MISC -D_POSIX_SOURCE -DNDEBUG

grugpu.o: grugpu.cu
	nvcc $(CUDAFLAGS) $(CUDA_INCLUDE_DIR) -c -O3 grugpu.cu -o grugpu.o

flappie : flappie.c nnfeatures.c flappie_common.c flappie_matrix.c flappie_output.c flappie_structures.c util.c fast5_interface.c flappie.c networks.c layers.c
	gcc $(CFLAGS) $(CUDA_LIB_DIR) $(utilS) $(CUDALIBS) nnfeatures.c flappie_common.c flappie_matrix.c flappie_output.c flappie_structures.c util.c fast5_interface.c flappie.c networks.c layers.c grugpu.o -o flappie
#flappie.o : flappie.c nnfeatures.c flappie_common.c flappie_matrix.c flappie_output.c flappie_structures.c util.c fast5_interface.c flappie.c networks.c layers.c
#gcc $(CFLAGS) -c flappie.c nnfeatures.c flappie_common.c flappie_matrix.c flappie_output.c flappie_structures.c util.c fast5_interface.c flappie.c networks.c layers.c -o flappie.o


#flappie: flappie.o grugpu.o
#	gcc $(CFLAGS) flappie.o grugpu.o $(CUDA_LIB_DIR) $(utilS) $(CUDALIBS) -o flappie

