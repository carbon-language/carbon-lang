// Check that ASan is able to print demangled frame name even w/o
// symbolization.

// RUN: %clangxx_asan -m64 -O0 %s -o %t && %t 2>&1 | FileCheck %s

#include <string.h>

namespace XXX {
struct YYY {
  static int ZZZ(int x) {
    char array[10];
    memset(array, 0, 10);
    return array[x];  // BOOOM
    // CHECK: {{ERROR: AddressSanitizer: stack-buffer-overflow}}
    // CHECK: {{READ of size 1 at 0x.* thread T0}}
    // CHECK: {{Address 0x.* is .* frame <XXX::YYY::ZZZ(.*)>}}
  }
};
}  // namespace XXX

int main(int argc, char **argv) {
  int res = XXX::YYY::ZZZ(argc + 10);
  return res;
}
