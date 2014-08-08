// RUN: %clang_cc1 -fprofile-instr-generate -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name builtinmacro.c %s | FileCheck %s

// Test the coverage mapping generation for built-in macroes.

const char *filename (const char *name) {
  static const char this_file[] = __FILE__;
  return this_file;
}

// CHECK: filename
// CHECK-NEXT: File 0, 5:41 -> 8:2 = #0 (HasCodeBefore = 0)
// CHECK-NEXT: main

int main() {
  filename(__FILE__ "test.c");
  return 0;
}
