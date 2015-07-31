// Simple test for a fuzzer. The fuzzer must find the interesting switch value.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <iostream>

static volatile int Sink;

template<class T>
bool Switch(const uint8_t *Data, size_t Size) {
  T X;
  if (Size < sizeof(X)) return false;
  memcpy(&X, Data, sizeof(X));
  switch (X) {
    case 1: Sink = __LINE__; break;
    case 101: Sink = __LINE__; break;
    case 1001: Sink = __LINE__; break;
    case 10001: Sink = __LINE__; break;
    case 100001: Sink = __LINE__; break;
    case 1000001: Sink = __LINE__; break;
    case 10000001: Sink = __LINE__; break;
    case 100000001: return true;
  }
  return false;
}

extern "C" void LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Switch<int>(Data, Size) && Size >= 12 &&
      Switch<uint64_t>(Data + 4, Size - 4)) {
    std::cout << "BINGO; Found the target, exiting\n";
    exit(1);
  }
}

