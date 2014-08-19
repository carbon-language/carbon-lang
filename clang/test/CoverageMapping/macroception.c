// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name macroception.c %s | FileCheck %s

#define M2 {
#define M1 M2
#define M22 }
#define M11 M22

                    // CHECK: main
                    // CHECK-NEXT: File 0, 3:12 -> 3:13 = #0 (HasCodeBefore = 0)
                    // CHECK-NEXT: Expansion,File 1, 4:12 -> 4:14 = #0 (HasCodeBefore = 0, Expanded file = 0)
int main() M1       // CHECK-NEXT: Expansion,File 2, [[@LINE]]:12 -> [[@LINE]]:14 = #0 (HasCodeBefore = 0, Expanded file = 1)
  return 0;         // CHECK-NEXT: File 2, [[@LINE]]:3 -> [[@LINE+1]]:2 = #0 (HasCodeBefore = 0)
}

                    // CHECK-NEXT: func2
void func2() {      // CHECK-NEXT: File 0, [[@LINE]]:14 -> [[@LINE+1]]:12 = #0 (HasCodeBefore = 0)
  int x = 0;
M11                 // CHECK-NEXT: Expansion,File 0, [[@LINE]]:1 -> [[@LINE]]:4 = #0 (HasCodeBefore = 0, Expanded file = 2)
                    // CHECK-NEXT: File 1, 5:13 -> 5:14 = #0 (HasCodeBefore = 0)
                    // CHECK-NEXT: Expansion,File 2, 6:13 -> 6:16 = #0 (HasCodeBefore = 0, Expanded file = 1)

                    // CHECK-NEXT: func3
                    // CHECK-NEXT: File 0, 3:12 -> 3:13 = #0 (HasCodeBefore = 0)
                    // CHECK-NEXT: Expansion,File 1, 4:12 -> 4:14 = #0 (HasCodeBefore = 0, Expanded file = 0)
void func3() M1     // CHECK-NEXT: Expansion,File 2, [[@LINE]]:14 -> [[@LINE]]:16 = #0 (HasCodeBefore = 0, Expanded file = 1)
  int x = 0;        // CHECK-NEXT: File 2, [[@LINE]]:3 -> [[@LINE]]:12 = #0 (HasCodeBefore = 0)
M11                 // CHECK-NEXT: Expansion,File 2, [[@LINE]]:1 -> [[@LINE]]:4 = #0 (HasCodeBefore = 0, Expanded file = 4)
                    // CHECK-NEXT: File 3, 5:13 -> 5:14 = #0 (HasCodeBefore = 0)
                    // CHECK-NEXT: Expansion,File 4, 6:13 -> 6:16 = #0 (HasCodeBefore = 0, Expanded file = 3)

                    // CHECK-NEXT: func4
                    // CHECK-NEXT: File 0, 3:12 -> 3:13 = #0 (HasCodeBefore = 0)
                    // CHECK-NEXT: Expansion,File 1, 4:12 -> 4:14 = #0 (HasCodeBefore = 0, Expanded file = 0)
                    // CHECK-NEXT: Expansion,File 2, [[@LINE+1]]:14 -> [[@LINE+1]]:16 = #0 (HasCodeBefore = 0, Expanded file = 1)
void func4() M1 M11 // CHECK-NEXT: Expansion,File 2, [[@LINE]]:17 -> [[@LINE]]:20 = #0 (HasCodeBefore = 0, Expanded file = 4)
                    // CHECK-NEXT: File 3, 5:13 -> 5:14 = #0 (HasCodeBefore = 0)
                    // CHECK-NEXT: Expansion,File 4, 6:13 -> 6:16 = #0 (HasCodeBefore = 0, Expanded file = 3)
