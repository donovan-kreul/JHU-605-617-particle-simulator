all: simulator.o bitmap.o
	nvcc -lcublas simulator.o bitmap.o -o simulator.exe -Wno-deprecated-gpu-targets

debug: simulator.o bitmap.o
	nvcc -lcublas -g -G simulator.o bitmap.o -o simulator.exe -Wno-deprecated-gpu-targets

simulator.o: 
	nvcc -c simulator.cu -o simulator.o

bitmap.o:
	nvcc -c bitmap.cpp -o bitmap.o

clean:
	rm -rf *.o *.exe *.bmp