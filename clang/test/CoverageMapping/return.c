// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name return.c %s | FileCheck %s

                                // CHECK: func
void func() {                   // CHECK: File 0, [[@LINE]]:13 -> [[@LINE+3]]:2 = #0 (HasCodeBefore = 0)
  return;
  int i = 0;                    // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:12 = 0 (HasCodeBefore = 0)
}

                                // CHECK-NEXT: func2
void func2() {                  // CHECK-NEXT: File 0, [[@LINE]]:14 -> [[@LINE+13]]:2 = #0 (HasCodeBefore = 0)
                                // CHECK-NEXT: File 0, [[@LINE+2]]:18 -> [[@LINE+2]]:24 = ((#0 + #1) - #2) (HasCodeBefore = 0)
                                // CHECK-NEXT: File 0, [[@LINE+1]]:26 -> [[@LINE+1]]:29 = (#1 - #2) (HasCodeBefore = 0)
  for(int i = 0; i < 10; ++i) { // CHECK-NEXT: File 0, [[@LINE]]:31 -> [[@LINE+9]]:4 = #1 (HasCodeBefore = 0)
    if(i > 2) {                 // CHECK-NEXT: File 0, [[@LINE]]:15 -> [[@LINE+2]]:6 = #2 (HasCodeBefore = 0)
      return;
    }                           // CHECK-NEXT: File 0, [[@LINE+1]]:5 -> [[@LINE+3]]:11 = (#1 - #2) (HasCodeBefore = 0)
    if(i == 3) {                // CHECK-NEXT: File 0, [[@LINE]]:16 -> [[@LINE+2]]:6 = #3 (HasCodeBefore = 0)
      int j = 1;
    } else {                    // CHECK-NEXT: File 0, [[@LINE]]:12 -> [[@LINE+2]]:6 = ((#1 - #2) - #3) (HasCodeBefore = 0)
      int j = 2;
    }
  }
}

                               // CHECK-NEXT: func3
void func3(int x) {            // CHECK-NEXT: File 0, [[@LINE]]:19 -> [[@LINE+9]]:2 = #0 (HasCodeBefore = 0)
  if(x > 5) {                  // CHECK-NEXT: File 0, [[@LINE]]:13 -> [[@LINE+6]]:4 = #1 (HasCodeBefore = 0)
    while(x >= 9) {            // CHECK-NEXT: File 0, [[@LINE]]:11 -> [[@LINE]]:17 = #1 (HasCodeBefore = 0)
      return;                  // CHECK-NEXT: File 0, [[@LINE-1]]:19 -> [[@LINE+2]]:6 = #2 (HasCodeBefore = 0)
      --x;                     // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:10 = 0 (HasCodeBefore = 0)
    }
    int i = 0;                 // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:14 = (#1 - #2) (HasCodeBefore = 0)
  }
  int j = 0;                   // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:12 = (#0 - #2) (HasCodeBefore = 0)
}
