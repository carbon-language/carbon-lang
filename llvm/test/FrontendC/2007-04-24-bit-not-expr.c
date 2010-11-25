// PR 1346
// RUN: %llvmgcc -S %s  -o /dev/null
extern bar(void *);

void f(void *cd) {
  bar(((void *)((unsigned long)(cd) ^ -1)));
}
