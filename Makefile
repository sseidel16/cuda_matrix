NVCC = nvcc
CUBLAS = -lcublas
TARGET = main
SOURCES = src/main.cu

all:
	$(NVCC) -arch=sm_86 $(SOURCES) -o $(TARGET) $(CUBLAS)

clean:
	rm -f $(TARGET)
