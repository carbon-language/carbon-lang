// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name casts.c %s | FileCheck %s

int main() {                                                   // CHECK: File 0, [[@LINE]]:12 -> [[@LINE+4]]:2 = #0 (HasCodeBefore = 0)
                                                               // CHECK-NEXT: File 0, [[@LINE+1]]:41 -> [[@LINE+1]]:54 = #1 (HasCodeBefore = 0)
  int window_size = (sizeof(int) <= 2 ? (unsigned)512 : 1024); // CHECK-NEXT: File 0, [[@LINE]]:57 -> [[@LINE]]:61 = (#0 - #1) (HasCodeBefore = 0)
  return 0;
}




