// Don't run through %symbolize to avoid c++filt demangling.
// RUN: %clangxx_asan -m64 -O0 %s -o %t && %t 2>&1 | FileCheck %s

namespace XXX {
class YYY {
 public:
  static int ZZZ[];
};
int YYY::ZZZ[] = {0, 1, 2, 3};
}

int main(int argc, char **argv) {
  return XXX::YYY::ZZZ[argc + 5];  // BOOM
  // CHECK: {{READ of size 4 at 0x.*}}
  // CHECK: {{0x.* is located 8 bytes to the right of global variable}}
  // CHECK: 'XXX::YYY::ZZZ' {{.*}} of size 16
}
