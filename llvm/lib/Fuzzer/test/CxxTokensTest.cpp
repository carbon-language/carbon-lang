// Simple test for a fuzzer. The fuzzer must find a sequence of C++ tokens.
#include <cstdint>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <iostream>

static void Found() {
  std::cout << "BINGO; Found the target, exiting\n";
  exit(1);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  // looking for "thread_local unsigned A;"
  if (Size < 24) return 0;
  if (0 == memcmp(&Data[0], "thread_local", 12))
    if (Data[12] == ' ')
      if (0 == memcmp(&Data[13], "unsigned", 8))
        if (Data[21] == ' ')
          if (Data[22] == 'A')
            if (Data[23] == ';')
              Found();
  return 0;
}

