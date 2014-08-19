// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name continue.c %s | FileCheck %s

int main() {                    // CHECK: File 0, [[@LINE]]:12 -> [[@LINE+21]]:2 = #0 (HasCodeBefore = 0)
  int j = 0;                    // CHECK-NEXT: File 0, [[@LINE+2]]:18 -> [[@LINE+2]]:24 = (#0 + #1) (HasCodeBefore = 0)
                                // CHECK-NEXT: File 0, [[@LINE+1]]:26 -> [[@LINE+1]]:29 = #1 (HasCodeBefore = 0)
  for(int i = 0; i < 20; ++i) { // CHECK-NEXT: File 0, [[@LINE]]:31 -> [[@LINE+17]]:4 = #1 (HasCodeBefore = 0)
    if(i < 10) {                // CHECK-NEXT: File 0, [[@LINE]]:16 -> [[@LINE+13]]:6 = #2 (HasCodeBefore = 0)
      if(i < 5) {               // CHECK-NEXT: File 0, [[@LINE]]:17 -> [[@LINE+3]]:8 = #3 (HasCodeBefore = 0)
        continue;
        j = 1;                   // CHECK-NEXT: File 0, [[@LINE]]:9 -> [[@LINE]]:14 = 0 (HasCodeBefore = 0)
      } else {                   // CHECK-NEXT: File 0, [[@LINE]]:14 -> [[@LINE+7]]:13 = (#2 - #3) (HasCodeBefore = 0)
        j = 2;
      }
      j = 3;
      if(i < 7) {                // CHECK-NEXT: File 0, [[@LINE]]:17 -> [[@LINE+3]]:8 = #4 (HasCodeBefore = 0)
        continue;
        j = 4;                   // CHECK-NEXT: File 0, [[@LINE]]:9 -> [[@LINE]]:14 = 0 (HasCodeBefore = 0)
      } else j = 5;              // CHECK-NEXT: File 0, [[@LINE]]:14 -> [[@LINE+1]]:12 = ((#2 - #3) - #4) (HasCodeBefore = 0)
      j = 6;
    } else                       // CHECK-NEXT: File 0, [[@LINE+1]]:7 -> [[@LINE+1]]:12 = (#1 - #2) (HasCodeBefore = 0)
      j = 7;
    j = 8;                       // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:10 = ((#1 - #3) - #4) (HasCodeBefore = 0)
  }
}
