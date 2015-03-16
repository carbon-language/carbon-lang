// Checks for reading various formats.

// CHECK: 100| [[@LINE+1]]|int main
int main(int argc, const char *argv[]) {}

// RUN: llvm-profdata merge %S/Inputs/binary-formats.proftext -o %t.profdata
// RUN: llvm-cov show %S/Inputs/binary-formats.macho32l -instr-profile %t.profdata -no-colors -filename-equivalence %s | FileCheck %s
// RUN: llvm-cov show %S/Inputs/binary-formats.macho64l -instr-profile %t.profdata -no-colors -filename-equivalence %s | FileCheck %s

// llvm-cov doesn't work on big endian yet
// XFAIL: powerpc64-, s390x, mips-, mips64-, sparc
