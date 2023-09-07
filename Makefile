CFLAGS = -O3 -march=native -ffast-math

matmul : matmul.cpp 
	g++ $(CFLAGS) -o $@ $<