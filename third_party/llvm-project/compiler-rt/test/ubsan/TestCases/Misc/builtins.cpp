// REQUIRES: arch=x86_64
//
// RUN: %clangxx -fsanitize=builtin -w %s -O3 -o %t
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=RECOVER
// RUN: %clangxx -fsanitize=builtin -fno-sanitize-recover=builtin -w %s -O3 -o %t.abort
// RUN: not %run %t.abort 2>&1 | FileCheck %s --check-prefix=ABORT

void check_ctz(int n) {
  // ABORT: builtins.cpp:[[@LINE+2]]:17: runtime error: passing zero to ctz(), which is not a valid argument
  // RECOVER: builtins.cpp:[[@LINE+1]]:17: runtime error: passing zero to ctz(), which is not a valid argument
  __builtin_ctz(n);

  // RECOVER: builtins.cpp:[[@LINE+1]]:18: runtime error: passing zero to ctz(), which is not a valid argument
  __builtin_ctzl(n);

  // RECOVER: builtins.cpp:[[@LINE+1]]:19: runtime error: passing zero to ctz(), which is not a valid argument
  __builtin_ctzll(n);
}

void check_clz(int n) {
  // RECOVER: builtins.cpp:[[@LINE+1]]:17: runtime error: passing zero to clz(), which is not a valid argument
  __builtin_clz(n);

  // RECOVER: builtins.cpp:[[@LINE+1]]:18: runtime error: passing zero to clz(), which is not a valid argument
  __builtin_clzl(n);

  // RECOVER: builtins.cpp:[[@LINE+1]]:19: runtime error: passing zero to clz(), which is not a valid argument
  __builtin_clzll(n);
}

int main() {
  check_ctz(0);
  check_clz(0);
  return 0;
}
