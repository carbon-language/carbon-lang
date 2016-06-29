// To create the covmapping for this file, copy this file to /tmp/dots/test.c,
// cd into /tmp/dots, and pass "../dots/double_dots.c" to the compiler. Use
// llvm-cov convert-for-testing to extract the covmapping.

// RUN: llvm-profdata merge %S/Inputs/double_dots.proftext -o %t.profdata
// RUN: llvm-cov show %S/Inputs/double_dots.covmapping -instr-profile=%t.profdata -o %t.dir
// RUN: FileCheck -input-file=%t.dir/index.txt %s

// CHECK-NOT: coverage{{.*}}dots{{.*}}..{{.*}}dots

int main() {}
