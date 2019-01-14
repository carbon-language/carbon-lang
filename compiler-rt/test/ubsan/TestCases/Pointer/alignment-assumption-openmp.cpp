// RUN: %clang   -x c   -fsanitize=alignment -fopenmp-simd -O0 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="error:"
// RUN: %clang   -x c   -fsanitize=alignment -fopenmp-simd -O1 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="error:"
// RUN: %clang   -x c   -fsanitize=alignment -fopenmp-simd -O2 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="error:"
// RUN: %clang   -x c   -fsanitize=alignment -fopenmp-simd -O3 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="error:"

// RUN: %clang   -x c++ -fsanitize=alignment -fopenmp-simd -O0 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="error:"
// RUN: %clang   -x c++ -fsanitize=alignment -fopenmp-simd -O1 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="error:"
// RUN: %clang   -x c++ -fsanitize=alignment -fopenmp-simd -O2 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="error:"
// RUN: %clang   -x c++ -fsanitize=alignment -fopenmp-simd -O3 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not=" assumption " --implicit-check-not="note:" --implicit-check-not="error:"

int main(int argc, char* argv[]) {
#pragma omp for simd aligned(argv : 0x40000000)
  for(int x = 0; x < 1; x++)
    argv[x] = argv[x];
  // CHECK: {{.*}}alignment-assumption-{{.*}}.cpp:[[@LINE-3]]:30: runtime error: assumption of 1073741824 byte alignment for pointer of type 'char **' failed
  // CHECK: 0x{{.*}}: note: address is {{.*}} aligned, misalignment offset is {{.*}} byte

  return 0;
}
