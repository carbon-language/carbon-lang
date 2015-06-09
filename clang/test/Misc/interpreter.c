// RUN: clang-interpreter %s | FileCheck %s
// REQUIRES: native, examples, shell

int printf(const char *, ...);

int main() {
  // CHECK: {{Hello world!}}
  printf("Hello world!\n");
  return 0;
}
