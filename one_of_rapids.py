import ctypes

def test_cuinit():
    cuda = ctypes.CDLL('libcuda.so')
    result = cuda.cuInit(0)
    if result == 0:
        print("cuInit successful")
    else:
        print(f"cuInit failed with error code {result}")

if __name__ == "__main__":
    test_cuinit()