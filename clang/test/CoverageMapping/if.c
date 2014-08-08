// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name if.c %s | FileCheck %s

int main() {
  int i = 0;
  if(i == 0) i = 1;
  if(i == 1)
    i = 2;
  if(i == 0) i = 1;
  if(i == 0)
    i = 1;
  if(i == 0) {
    i = 1;
  }
  if(i == 0) { i = 1;
    i = 2;
  }
  if(i != 0) {
    i = 1;
  } else {
    i = 3;
  }
  i = i == 0?
        i + 1 :
        i + 2;
  i = i == 0?i + 12:i + 10;
  i = i < 20?i + 13:i + 20;

  for(int j = 0; j < 10; ++j) {
    if(j < 3) {
      i = 2;
    } else
      i = 3;
    if(j < 4) i = 0; else i = 1;
    if(j < 0) i = 0; else i = 1;
    if(j < 0) ; else i = 1;
  }
  return 0;
}

// CHECK: File 0, 3:12 -> 38:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 5:14 -> 5:19 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 7:5 -> 7:10 = #2 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 8:14 -> 8:19 = #3 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 10:5 -> 10:10 = #4 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 11:14 -> 13:4 = #5 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 14:14 -> 16:4 = #6 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 17:14 -> 19:4 = #7 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 19:10 -> 21:4 = (#0 - #7) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 23:9 -> 23:14 = #8 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 24:9 -> 24:14 = (#0 - #8) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 25:14 -> 25:20 = #9 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 25:21 -> 25:27 = (#0 - #9) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 26:14 -> 26:20 = #10 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 26:21 -> 26:27 = (#0 - #10) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 28:18 -> 28:24 = (#0 + #11) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 28:26 -> 28:29 = #11 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 28:31 -> 36:4 = #11 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 29:15 -> 31:6 = #12 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 32:7 -> 32:12 = (#11 - #12) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 33:15 -> 33:20 = #13 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 33:27 -> 33:32 = (#11 - #13) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 34:15 -> 34:20 = #14 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 34:27 -> 34:32 = (#11 - #14) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 35:15 -> 35:16 = #15 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 35:22 -> 35:27 = (#11 - #15) (HasCodeBefore = 0)
