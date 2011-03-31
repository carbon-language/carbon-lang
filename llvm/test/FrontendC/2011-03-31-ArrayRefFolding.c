// RUN: %llvmgcc -S -o - -m32 -Os %s | FileCheck %s
// PR9571

struct t {
  int x;
};

extern struct t *cfun;

int f(void) {
  if (!(cfun + 0))
// CHECK: icmp eq %struct.t* %0, null
    return 0;
  return cfun->x;
}
