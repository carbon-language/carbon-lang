// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name macros.c %s | FileCheck %s

void bar();
#define MACRO return; bar()
#define MACRO_2 bar()
#define MACRO_1 return; MACRO_2

void func() {
  int i = 0;
  MACRO;
  i = 2;
}

// CHECK: func
// CHECK-NEXT: File 0, 8:13 -> 12:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 0, 10:3 -> 10:8 = #0 (HasCodeBefore = 0, Expanded file = 1)
// CHECK-NEXT: File 0, 11:3 -> 11:8 = 0 (HasCodeBefore = 0)
// CHECK-NEXT: File 1, 4:15 -> 4:21 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 1, 4:23 -> 4:28 = 0 (HasCodeBefore = 0)

void func2() {
  int i = 0;
  MACRO_1;
  i = 2;
}

// CHECK-NEXT: func2
// CHECK-NEXT: File 0, 21:14 -> 25:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 0, 23:3 -> 23:10 = #0 (HasCodeBefore = 0, Expanded file = 1)
// CHECK-NEXT: File 0, 24:3 -> 24:8 = 0 (HasCodeBefore = 0)
// CHECK-NEXT: File 1, 6:17 -> 6:23 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 1, 6:25 -> 6:32 = 0 (HasCodeBefore = 0, Expanded file = 2)
// CHECK-NEXT: File 2, 5:17 -> 5:22 = 0 (HasCodeBefore = 0)

int main() {
  func();
  func2();
  return 0;
}

void bar() {
}
