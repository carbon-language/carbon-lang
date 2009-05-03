// RUN: clang-cc %s -E | grep "movl %eax"

#define R1E %eax
#define epilogue(r1) movl r1;
epilogue(R1E)
