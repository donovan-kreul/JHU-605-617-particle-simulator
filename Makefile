all: simulator.o bitmap.o args.o
	nvcc -lcurand simulator.o bitmap.o args.o -o simulator -Wno-deprecated-gpu-targets

debug: clean simulator.o bitmap.o args.o
	nvcc -lcurand -g -G simulator.o bitmap.o args.o -o simulator -Wno-deprecated-gpu-targets

simulator.o: 
	nvcc -c simulator.cu -o simulator.o -Wno-deprecated-gpu-targets

bitmap.o:
	nvcc -c bitmap.cpp -o bitmap.o -Wno-deprecated-gpu-targets

args.o:
	nvcc -c args.cpp -o args.o -Wno-deprecated-gpu-targets

clean:
	rm -rf *.o simulator ./img/*.bmp ./img/*.mp4