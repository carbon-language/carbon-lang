// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name casts.c %s | FileCheck %s

int main() {
  int window_size = (sizeof(int) <= 2 ? (unsigned)512 : 1024);
  return 0;
}

// CHECK: File 0, 3:12 -> 6:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 4:41 -> 4:54 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 4:57 -> 4:61 = (#0 - #1) (HasCodeBefore = 0)
