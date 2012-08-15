// RUN: %clangxx_asan -m64 -O2 %s -o %t
// RUN: %t 2>&1 | %symbolizer | FileCheck %s

#include <string.h>
int main(int argc, char **argv) {
  char x[10];
  memset(x, 0, 10);
  int res = x[argc * 10];  // BOOOM
  // CHECK: {{READ of size 1 at 0x.* thread T0}}
  // CHECK: {{    #0 0x.* in main .*stack-overflow.cc:8}}
  // CHECK: {{Address 0x.* is .* frame <main>}}
  return res;
}
