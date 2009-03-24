// RUN: clang-cc -emit-llvm %s -o %t &&
// RUN: grep 'internal constant \[10 x i8\]' %t &&
// RUN: not grep -F "[5 x i8]" %t &&
// RUN: not grep "store " %t

void test(void) {
  char a[10] = "asdf";
  char b[10] = { "asdf" };
}

