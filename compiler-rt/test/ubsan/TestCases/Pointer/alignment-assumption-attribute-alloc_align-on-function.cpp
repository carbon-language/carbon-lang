// FIXME: https://code.google.com/p/address-sanitizer/issues/detail?id=316
// I'm not sure this is actually *that* issue, but this seems oddly similar to the other XFAIL'ed cases.
// UNSUPPORTED: android
// UNSUPPORTED: ios

// RUN: %clang   -x c   -fsanitize=alignment -O0 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="error:"
// RUN: %clang   -x c   -fsanitize=alignment -O1 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="error:"
// RUN: %clang   -x c   -fsanitize=alignment -O2 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="error:"
// RUN: %clang   -x c   -fsanitize=alignment -O3 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="error:"

// RUN: %clang   -x c++ -fsanitize=alignment -O0 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="error:"
// RUN: %clang   -x c++ -fsanitize=alignment -O1 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="error:"
// RUN: %clang   -x c++ -fsanitize=alignment -O2 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="error:"
// RUN: %clang   -x c++ -fsanitize=alignment -O3 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="error:"

char **__attribute__((alloc_align(2)))
passthrough(char **x, unsigned long alignment) {
  return x;
}

int main(int argc, char* argv[]) {
  passthrough(argv, 0x80000000);
  // CHECK: {{.*}}alignment-assumption-{{.*}}.cpp:[[@LINE-1]]:3: runtime error: assumption of 2147483648 byte alignment for pointer of type 'char **' failed
  // CHECK: {{.*}}alignment-assumption-{{.*}}.cpp:[[@LINE-8]]:23: note: alignment assumption was specified here
  // CHECK: 0x{{.*}}: note: address is {{.*}} aligned, misalignment offset is {{.*}} byte

  return 0;
}
