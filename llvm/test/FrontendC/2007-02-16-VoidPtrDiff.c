// RUN: %llvmgcc %s -S -o -

void foo(void *ptr, int test) {
  (ptr - ((void *) test + 0x2000));
}
