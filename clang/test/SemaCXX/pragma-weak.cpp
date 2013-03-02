// RUN: %clang_cc1 %s

#pragma weak foo
static void foo();
extern "C" {
  void foo() {
  };
}
