// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s
// rdar: // 8353567
// pr7726

extern "C" int printf(...);

int main(int argc, char **argv) {
// CHECK: phi i8* [ inttoptr (i64 3735928559 to i8*),
    printf("%p\n", (void *)0xdeadbeef ? : (void *)0xaaaaaa);
    return 0;
}
