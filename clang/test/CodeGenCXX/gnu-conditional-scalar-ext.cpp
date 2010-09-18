// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s
// rdar: // 8353567
// pr7726

extern "C" int printf(...);

int main(int argc, char **argv) {
// CHECK: phi i8* [ inttoptr (i64 3735928559 to i8*),
    printf("%p\n", (void *)0xdeadbeef ? : (void *)0xaaaaaa);
    return 0;
}

// rdar://8446940
namespace radar8446940 {
extern "C" void abort();

int main () {
  char x[1];
  char *y = x ? : 0;

  if (x != y)
    abort();
}
}
