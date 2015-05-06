// Simple test for a fuzzer. The fuzzer must find a particular string.
#include <cstring>
#include <cstdint>

extern "C" void LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size >= 10 && memcmp(Data, "0123456789", 10) == 0)
    __builtin_trap();
}
