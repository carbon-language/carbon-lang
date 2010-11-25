// RUN: %llvmgcc %s -O3 -S -o -
// PR1175

struct empty { };

void foo(struct empty *p) {
   p++;
}

