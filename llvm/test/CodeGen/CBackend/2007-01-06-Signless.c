// RUN: %llvmgcc -S %s -o - | llvm-as | llc -march=c | \
// RUN:  grep {(unsigned short}

int Z = -1;

int test(unsigned short X, short Y) { return X+Y+Z; }
