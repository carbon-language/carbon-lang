// RUN: clang -emit-llvm %s -o - | not grep "[5 x i8]"
// RUN: clang -emit-llvm %s -o - | not grep "store"

void test(void) {
  char a[10] = "asdf";
  char b[10] = { "asdf" };
}

