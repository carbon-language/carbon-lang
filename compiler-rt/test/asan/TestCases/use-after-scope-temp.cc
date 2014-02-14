// RUN: %clangxx_asan -O0 -fsanitize=use-after-scope %s -o %t && \
// RUN:     %t 2>&1 | FileCheck %s
//
// Lifetime for temporaries is not emitted yet.
// XFAIL: *

#include <stdio.h>

struct IntHolder {
  explicit IntHolder(int val) : val(val) {
    printf("IntHolder: %d\n", val);
  }
  int val;
};

const IntHolder *saved;

void save(const IntHolder &holder) {
  saved = &holder;
}

int main(int argc, char *argv[]) {
  save(IntHolder(10));
  int x = saved->val;  // BOOM
  // CHECK: ERROR: AddressSanitizer: stack-use-after-scope
  // CHECK:  #0 0x{{.*}} in main {{.*}}use-after-scope-temp.cc:[[@LINE-2]]
  printf("saved value: %d\n", x);
  return 0;
}
