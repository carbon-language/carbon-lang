// RUN: %llvmgcc -S %s -o -
// PR4214
typedef void vt;
void (*func_ptr)(vt my_vt);
