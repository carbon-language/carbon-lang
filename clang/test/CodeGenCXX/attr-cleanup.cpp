// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

namespace N {
  void free(void *i) {}
}

int main(void) {
  // CHECK: call void @_ZN1N4freeEPv(i8* %0)
  void *fp __attribute__((cleanup(N::free)));
  return 0;
}
