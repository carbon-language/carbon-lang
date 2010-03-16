// RUN: %llvmgxx -S %s -o - | FileCheck %s

#include <utility>

typedef std::pair<int,int> P;
// CHECK: @_ZZ1fvE1X {{.*}} undef

P f() {
  static const P X = P(1,2);
  return X;
}
