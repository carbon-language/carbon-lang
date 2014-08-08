// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name test.c %s | FileCheck %s

void bar();
static void static_func();

int main() {
  for(int i = 0; i < 10; ++i) {
    bar();
  }
  static_func();
  return 0;
}

// CHECK: main
// CHECK-NEXT: File 0, 6:12 -> 12:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 7:18 -> 7:24 = (#0 + #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 7:26 -> 7:29 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 7:31 -> 9:4 = #1 (HasCodeBefore = 0)

void foo() {
  if(1) {
    int i = 0;
  }
}

// CHECK-NEXT: foo
// CHECK-NEXT: File 0, 20:12 -> 24:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 21:9 -> 23:4 = #1 (HasCodeBefore = 0)

void bar() {
}

// CHECK-NEXT: bar
// CHECK-NEXT: File 0, 30:12 -> 31:2 = #0 (HasCodeBefore = 0)

void static_func() { }

// CHECK-NEXT: static_func
// CHECK: File 0, 36:20 -> 36:23 = #0 (HasCodeBefore = 0)

static void func() { }

// CHECK-NEXT: func
// CHECK: File 0, 41:20 -> 41:23 = 0 (HasCodeBefore = 0)
