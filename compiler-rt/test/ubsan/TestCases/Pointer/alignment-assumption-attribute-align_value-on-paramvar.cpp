// RUN: %clang   -x c   -fsanitize=alignment -O0 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="error:"
// RUN: %clang   -x c   -fsanitize=alignment -O1 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="error:"
// RUN: %clang   -x c   -fsanitize=alignment -O2 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="error:"
// RUN: %clang   -x c   -fsanitize=alignment -O3 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="error:"

// RUN: %clang   -x c++ -fsanitize=alignment -O0 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="error:"
// RUN: %clang   -x c++ -fsanitize=alignment -O1 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="error:"
// RUN: %clang   -x c++ -fsanitize=alignment -O2 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="error:"
// RUN: %clang   -x c++ -fsanitize=alignment -O3 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="error:"

char** passthrough(__attribute__((align_value(0x80000000))) char **x) {
  return x;
}

int main(int argc, char* argv[]) {
  passthrough(argv);
  // CHECK: {{.*}}alignment-assumption-{{.*}}.cpp:[[@LINE-5]]:10: runtime error: assumption of 2147483648 byte alignment for pointer of type 'char **' failed
  // CHECK: {{.*}}alignment-assumption-{{.*}}.cpp:[[@LINE-7]]:35: note: alignment assumption was specified here
  // CHECK: 0x{{.*}}: note: address is {{.*}} aligned, misalignment offset is {{.*}} byte

  // FIXME: shouldn't there be an assumption on the caller's side too?

  return 0;
}
