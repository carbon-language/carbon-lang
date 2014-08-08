// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name preprocessor.c %s | FileCheck %s

void func() {
  int i = 0;
#ifdef MACRO
  int x = i;
#endif
}

// CHECK: func
// CHECK: File 0, 3:13 -> 8:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: Skipped,File 0, 5:2 -> 7:2 = 0 (HasCodeBefore = 0)

#if 0
  int g = 0;

  void bar() { }
#endif

int main() {
  int i = 0;
#if 0
  if(i == 0) {
    i = 1;
  }
#endif

#if 1
  if(i == 0) {
    i = 1;
  }
#else
  if(i == 1) {
    i = 0;
  }
}
#endif
  return 0;
}

// CHECK: main
// CHECK-NEXT: File 0, 20:12 -> 39:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: Skipped,File 0, 22:2 -> 26:2 = 0 (HasCodeBefore = 0)
// CHECK-NEXT: File 0, 29:14 -> 31:4 = #1 (HasCodeBefore = 0)
// CHECK-NEXT: Skipped,File 0, 32:2 -> 37:2 = 0 (HasCodeBefore = 0)
