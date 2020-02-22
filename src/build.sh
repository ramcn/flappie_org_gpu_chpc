make -f makefile.cuda

gcc -o flappie -O3 -I/usr/include/hdf5/serial  -I/usr/local/cuda-10.2/include -Wno-sign-compare -march=native  -Wno-format  -DUSE_SSE2  -D__USE_MISC -D_POSIX_SOURCE -DNDEBUG -o flappie decode.c nnfeatures.c flappie_common.c flappie_matrix.c flappie_output.c flappie_structures.c util.c fast5_interface.c flappie.c networks.c layers.c grugpu.o -L/usr/local/cuda-10.2/lib64 -lhdf5_serial -lpthread -lblas -lcublas -lcudart -lm -lstdc++
