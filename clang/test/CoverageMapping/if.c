// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name if.c %s | FileCheck %s

int main() {                    // CHECK: File 0, [[@LINE]]:12 -> [[@LINE+21]]:2 = #0 (HasCodeBefore = 0)
  int i = 0;
  if(i == 0) i = 1;             // CHECK-NEXT: File 0, [[@LINE]]:14 -> [[@LINE]]:19 = #1 (HasCodeBefore = 0)
  if(i == 1)
    i = 2;                      // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:10 = #2 (HasCodeBefore = 0)
  if(i == 0) { i = 1;           // CHECK-NEXT: File 0, [[@LINE]]:14 -> [[@LINE+2]]:4 = #3 (HasCodeBefore = 0)
    i = 2;
  }
  if(i != 0) {                  // CHECK-NEXT: File 0, [[@LINE]]:14 -> [[@LINE+2]]:4 = #4 (HasCodeBefore = 0)
    i = 1;
  } else {                      // CHECK-NEXT: File 0, [[@LINE]]:10 -> [[@LINE+2]]:4 = (#0 - #4) (HasCodeBefore = 0)
    i = 3;
  }

  i = i == 0?
        i + 1 :                 // CHECK-NEXT: File 0, [[@LINE]]:9 -> [[@LINE]]:14 = #5 (HasCodeBefore = 0)
        i + 2;                  // CHECK-NEXT: File 0, [[@LINE]]:9 -> [[@LINE]]:14 = (#0 - #5) (HasCodeBefore = 0)
                                // CHECK-NEXT: File 0, [[@LINE+1]]:14 -> [[@LINE+1]]:20 = #6 (HasCodeBefore = 0)
  i = i == 0?i + 12:i + 10;     // CHECK-NEXT: File 0, [[@LINE]]:21 -> [[@LINE]]:27 = (#0 - #6) (HasCodeBefore = 0)

  return 0;
}
