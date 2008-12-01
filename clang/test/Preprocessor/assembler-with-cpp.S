// RUN: clang -E %s &&
// RUN: not clang -x c -E %s

#ifndef __ASSEMBLER__
#error "__ASSEMBLER__ not defined"
#endif
