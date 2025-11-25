NVCC = nvcc
TARGET = main
SOURCES = src/main.cu

all:
	$(NVCC) -arch=sm_86 $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)
