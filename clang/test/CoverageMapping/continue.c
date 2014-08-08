// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name continue.c %s | FileCheck %s

int main() {
  int j = 0;
  for(int i = 0; i < 20; ++i) {
    if(i < 10) {
      if(i < 5) {
        continue;
        j = 1;
      } else {
        j = 2;
      }
      j = 3;
      if(i < 7) {
        continue;
        j = 4;
      } else j = 5;
      j = 6;
    } else
      j = 7;
    j = 8;
  }
}

// CHECK: File 0, 3:12 -> 23:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 5:18 -> 5:24 = (#0 + #1) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 5:26 -> 5:29 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 5:31 -> 22:4 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 6:16 -> 19:6 = #2 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 7:17 -> 10:8 = #3 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 9:9 -> 9:14 = 0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 10:14 -> 17:13 = (#2 - #3) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 14:17 -> 17:8 = #4 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 16:9 -> 16:14 = 0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 17:14 -> 18:12 = ((#2 - #3) - #4) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 20:7 -> 20:12 = (#1 - #2) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 21:5 -> 21:10 = ((#1 - #3) - #4) (HasCodeBefore = 0)
