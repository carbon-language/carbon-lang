// RUN: %llvmgcc -O3 -S -o - %s | grep {volatile store}
// PR1352

struct foo {
  int x;
};

void copy(volatile struct foo *p, struct foo *q) {
  *p = *q;
}
