// RUN: %clang_cc1 -fblocks -fsyntax-only -Wunused-parameter %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fblocks -fsyntax-only -Wunused %s 2>&1 | FileCheck -check-prefix=CHECK-unused %s

int f0(int x,
       int y,
       int z __attribute__((unused))) {
  return x;
}

void f1() {
  (void)^(int x,
          int y,
          int z __attribute__((unused))) { return x; };
}

// Used when testing '-Wunused' to see that we only emit one diagnostic, and no
// warnings for the above cases.
static void achor() {};

// CHECK: 5:12: warning: unused parameter 'y'
// CHECK: 12:15: warning: unused parameter 'y'
// CHECK-unused: 1 warning generated