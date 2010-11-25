// RUN: %llvmgcc %s -S -O0 -o -
// RUN: %llvmgcc %s -S -O1 -o -
// rdar://6518089

static int bar();
void foo() {
  int a = bar();
}
int bar(unsigned a) {
}
