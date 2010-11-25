// RUN: %llvmgcc -xc %s -S -o - | grep -v div

int Diff(int *P, int *Q) { return P-Q; }
