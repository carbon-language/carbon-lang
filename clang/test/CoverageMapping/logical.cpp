// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name logical.cpp %s | FileCheck %s

int main() {
  bool bt = true;
  bool bf = false;
  bool a = bt && bf;
  a = bt &&
      bf;
  a = bf && bt;
  a = bf &&
      bt;
  a = bf || bt;
  a = bf ||
      bt;
  a = bt || bf;
  a = bt ||
      bf;
  for(int j = 0; j < 10; ++j) {
    if(j < 2 && j < 6) a = true;
    a = j < 0 && j > 10;
    if(j < 0 && j > 10) a = false;
    a = j < 10 || j < 20;
  }
  return 0;
}

// CHECK: File 0, 3:12 -> 25:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 6:18 -> 6:20 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 8:7 -> 8:9 = #2 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 9:13 -> 9:15 = #3 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 11:7 -> 11:9 = #4 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 12:13 -> 12:15 = #5 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 14:7 -> 14:9 = #6 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 15:13 -> 15:15 = #7 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 17:7 -> 17:9 = #8 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 18:18 -> 18:24 = (#0 + #9) (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 18:26 -> 18:29 = #9 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 18:31 -> 23:4 = #9 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 19:17 -> 19:22 = #11 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 19:24 -> 19:32 = #10 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 20:18 -> 20:24 = #12 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 21:17 -> 21:23 = #14 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 21:25 -> 21:34 = #13 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 22:19 -> 22:25 = #15 (HasCodeBefore = 0)
