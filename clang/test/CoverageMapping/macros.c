// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name macros.c %s | FileCheck %s

void bar();
#define MACRO return; bar()
#define MACRO_2 bar()
#define MACRO_1 return; MACRO_2

               // CHECK: func
void func() {  // CHECK-NEXT: File 0, [[@LINE]]:13 -> [[@LINE+4]]:2 = #0 (HasCodeBefore = 0)
  int i = 0;
  MACRO;       // CHECK-NEXT: Expansion,File 0, [[@LINE]]:3 -> [[@LINE]]:8 = #0 (HasCodeBefore = 0, Expanded file = 1)
  i = 2;       // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:8 = 0 (HasCodeBefore = 0)
}
// CHECK-NEXT: File 1, 4:15 -> 4:21 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 1, 4:23 -> 4:28 = 0 (HasCodeBefore = 0)

               // CHECK-NEXT: func2
void func2() { // CHECK-NEXT: File 0, [[@LINE]]:14 -> [[@LINE+4]]:2 = #0 (HasCodeBefore = 0)
  int i = 0;
  MACRO_1;     // CHECK-NEXT: Expansion,File 0, [[@LINE]]:3 -> [[@LINE]]:10 = #0 (HasCodeBefore = 0, Expanded file = 1)
  i = 2;       // CHECK-NEXT: File 0, [[@LINE]]:3 -> [[@LINE]]:8 = 0 (HasCodeBefore = 0)
}
// CHECK-NEXT: File 1, 6:17 -> 6:23 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: Expansion,File 1, 6:25 -> 6:32 = 0 (HasCodeBefore = 0, Expanded file = 2)
// CHECK-NEXT: File 2, 5:17 -> 5:22 = 0 (HasCodeBefore = 0)

