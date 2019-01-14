// RUN: %clang -fsanitize=alignment -fno-sanitize-recover=alignment                           -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption "

// RUN: rm -f %tmp
// RUN: echo "[alignment]" >> %tmp
// RUN: echo "fun:main" >> %tmp
// RUN: %clang -fsanitize=alignment -fno-sanitize-recover=alignment -fsanitize-blacklist=%tmp -O0 %s -o %t && %run %t 2>&1

int main(int argc, char* argv[]) {
  __builtin_assume_aligned(argv, 0x80000000);
  // CHECK: {{.*}}alignment-assumption-blacklist.cpp:[[@LINE-1]]:28: runtime error: assumption of 2147483648 byte alignment for pointer of type 'char **' failed
  // CHECK: 0x{{.*}}: note: address is {{.*}} aligned, misalignment offset is {{.*}} byte

  return 0;
}
