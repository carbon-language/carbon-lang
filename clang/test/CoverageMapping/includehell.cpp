// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name includehell.cpp %s | FileCheck %s

#define CODE \
  x = x;\
  if (x == 0) {\
    x = 1;\
  } else {\
    x = 2;\
  }\
  if (true) {\
    x = x;\
  } else { \
    x = x; \
  }

int main() {
  int x = 0;
  #include "Inputs/code.h"
#include "Inputs/code.h"
  x = 0;
  CODE
  x = 0;
  CODE CODE
  if (false) {
    x = 0; CODE
  }
  return 0;
}

// CHECK: File 0, 1:1 -> 9:7 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 2:13 -> 4:2 = #3 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 4:8 -> 6:2 = (#0 - #3) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 7:11 -> 9:2 = #4 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 9:8 -> 11:2 = (#0 - #4) (HasCodeBefore = 0)
// CHECK-NEXT: File 1, 16:12 -> 28:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 1, 18:12 -> 18:27 = #0 (HasCodeBefore = 0, Expanded file = 2)
// CHECK-NEXT: Expansion,File 1, 19:10 -> 19:25 = #0 (HasCodeBefore = 0, Expanded file = 0)
// CHECK-NEXT: Expansion,File 1, 21:3 -> 21:7 = #0 (HasCodeBefore = 0, Expanded file = 3)
// CHECK-NEXT: Expansion,File 1, 23:3 -> 23:7 = #0 (HasCodeBefore = 0, Expanded file = 5)
// CHECK-NEXT: Expansion,File 1, 23:8 -> 23:12 = #0 (HasCodeBefore = 0, Expanded file = 4)
// CHECK-NEXT: File 1, 24:14 -> 26:4 = #11 (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 1, 25:12 -> 25:16 = #11 (HasCodeBefore = 0, Expanded file = 6)
// CHECK-NEXT: File 2, 1:1 -> 9:7 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 2, 2:13 -> 4:2 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 2, 4:8 -> 6:2 = (#0 - #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 2, 7:11 -> 9:2 = #2 (HasCodeBefore = 0)
// CHECK-NEXT: File 2, 9:8 -> 11:2 = (#0 - #2) (HasCodeBefore = 0)
// CHECK-NEXT: File 3, 4:3 -> 12:9 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 3, 5:15 -> 7:4 = #5 (HasCodeBefore = 0)
// CHECK-NEXT: File 3, 7:10 -> 9:4 = (#0 - #5) (HasCodeBefore = 0)
// CHECK-NEXT: File 3, 10:13 -> 12:4 = #6 (HasCodeBefore = 0)
// CHECK-NEXT: File 3, 12:10 -> 14:4 = (#0 - #6) (HasCodeBefore = 0)
// CHECK-NEXT: File 4, 4:3 -> 12:9 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 4, 5:15 -> 7:4 = #9 (HasCodeBefore = 0)
// CHECK-NEXT: File 4, 7:10 -> 9:4 = (#0 - #9) (HasCodeBefore = 0)
// CHECK-NEXT: File 4, 10:13 -> 12:4 = #10 (HasCodeBefore = 0)
// CHECK-NEXT: File 4, 12:10 -> 14:4 = (#0 - #10) (HasCodeBefore = 0)
// CHECK-NEXT: File 5, 4:3 -> 12:9 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 5, 5:15 -> 7:4 = #7 (HasCodeBefore = 0)
// CHECK-NEXT: File 5, 7:10 -> 9:4 = (#0 - #7) (HasCodeBefore = 0)
// CHECK-NEXT: File 5, 10:13 -> 12:4 = #8 (HasCodeBefore = 0)
// CHECK-NEXT: File 5, 12:10 -> 14:4 = (#0 - #8) (HasCodeBefore = 0)
// CHECK-NEXT: File 6, 4:3 -> 12:9 = #11 (HasCodeBefore = 0)
// CHECK-NEXT: File 6, 5:15 -> 7:4 = #12 (HasCodeBefore = 0)
// CHECK-NEXT: File 6, 7:10 -> 9:4 = (#11 - #12) (HasCodeBefore = 0)
// CHECK-NEXT: File 6, 10:13 -> 12:4 = #13 (HasCodeBefore = 0)
// CHECK-NEXT: File 6, 12:10 -> 14:4 = (#11 - #13) (HasCodeBefore = 0)
