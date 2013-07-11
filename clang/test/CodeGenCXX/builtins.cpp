// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

// PR8839
extern "C" char memmove();

int main() {
  // CHECK: call {{signext i8|i8}} @memmove()
  return memmove();
}

struct S;
// CHECK: define {{.*}} @_Z9addressofbR1SS0_(
S *addressof(bool b, S &s, S &t) {
  // CHECK: %[[LVALUE:.*]] = phi
  // CHECK: ret {{.*}}* %[[LVALUE]]
  return __builtin_addressof(b ? s : t);
}
