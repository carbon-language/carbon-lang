// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name macroception.c %s | FileCheck %s

#define M2 {
#define M1 M2
#define M22 }
#define M11 M22

int main() M1
  return 0;
}

// CHECK: main
// CHECK-NEXT: File 0, 3:12 -> 3:13 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 1, 4:12 -> 4:14 = #0 (HasCodeBefore = 0, Expanded file = 0)
// CHECK-NEXT: Expansion,File 2, 8:12 -> 8:14 = #0 (HasCodeBefore = 0, Expanded file = 1)
// CHECK-NEXT: File 2, 9:3 -> 10:2 = #0 (HasCodeBefore = 0)

void func2() {
  int x = 0;
M11

// CHECK-NEXT: func2
// CHECK-NEXT: File 0, 18:14 -> 19:12 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 0, 20:1 -> 20:4 = #0 (HasCodeBefore = 0, Expanded file = 2)
// CHECK-NEXT: File 1, 5:13 -> 5:14 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 2, 6:13 -> 6:16 = #0 (HasCodeBefore = 0, Expanded file = 1)

void func3() M1
  int x = 0;
M11

// CHECK-NEXT: func3
// CHECK-NEXT: File 0, 3:12 -> 3:13 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 1, 4:12 -> 4:14 = #0 (HasCodeBefore = 0, Expanded file = 0)
// CHECK-NEXT: Expansion,File 2, 28:14 -> 28:16 = #0 (HasCodeBefore = 0, Expanded file = 1)
// CHECK-NEXT: File 2, 29:3 -> 29:12 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 2, 30:1 -> 30:4 = #0 (HasCodeBefore = 0, Expanded file = 4)
// CHECK-NEXT: File 3, 5:13 -> 5:14 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 4, 6:13 -> 6:16 = #0 (HasCodeBefore = 0, Expanded file = 3)

void func4() M1 M11

// CHECK-NEXT: func4
// CHECK-NEXT: File 0, 3:12 -> 3:13 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 1, 4:12 -> 4:14 = #0 (HasCodeBefore = 0, Expanded file = 0)
// CHECK-NEXT: Expansion,File 2, 41:14 -> 41:16 = #0 (HasCodeBefore = 0, Expanded file = 1)
// CHECK-NEXT: Expansion,File 2, 41:17 -> 41:20 = #0 (HasCodeBefore = 0, Expanded file = 4)
// CHECK-NEXT: File 3, 5:13 -> 5:14 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 4, 6:13 -> 6:16 = #0 (HasCodeBefore = 0, Expanded file = 3)
