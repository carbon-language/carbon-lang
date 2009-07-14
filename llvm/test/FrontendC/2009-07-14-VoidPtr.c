// RUN: %llvmgcc -S %s -o -
// PR4556

extern void foo;
void *bar = &foo;

