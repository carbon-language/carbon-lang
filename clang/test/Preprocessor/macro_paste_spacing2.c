// RUN: clang-cc %s -E | grep "movl %eax"
// PR4132
#define R1E %eax
#define epilogue(r1) movl r1 ## E;
epilogue(R1)

