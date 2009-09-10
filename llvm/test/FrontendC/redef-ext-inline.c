// RUN: %llvmgcc -S %s -o -
// rdar://7208839

extern inline int f1 (void) {return 1;}
int f3 (void) {return f1();}
int f1 (void) {return 0;}
