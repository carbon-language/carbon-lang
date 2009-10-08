// RUN: clang-cc -triple i686-unknown-unknown -emit-llvm -o %t %s &&
// RUN: not grep 'ssp' %t &&
// RUN: clang-cc -triple i686-apple-darwin9 -emit-llvm -o %t %s &&
// RUN: not grep 'ssp' %t &&
// RUN: clang-cc -triple i686-apple-darwin10 -emit-llvm -o %t %s &&
// RUN: grep 'ssp' %t &&
// RUN: clang -fstack-protector-all -emit-llvm -S -o %t %s &&
// RUN: grep 'sspreq' %t &&
// RUN: clang -fstack-protector -emit-llvm -S -o %t %s &&
// RUN: grep 'ssp' %t &&
// RUN: clang -fno-stack-protector -emit-llvm -S -o %t %s &&
// RUN: not grep 'ssp' %t &&
// RUN: true

int printf(const char * _Format, ...);

void test1(const char *msg) {
  char a[strlen(msg) + 1];
  strcpy(a, msg);
  printf("%s\n", a);
}
