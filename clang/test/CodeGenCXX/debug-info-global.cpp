// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -emit-llvm -g %s -o - | FileCheck %s

// Multiple references to the same constant should result in only one entry in
// the globals list.

namespace ns {
const int cnst = 42;
}
int f1() {
  return ns::cnst + ns::cnst;
}

// CHECK: !MDCompileUnit(
// CHECK-SAME:           globals: [[GLOBALS:![0-9]*]]

// CHECK: [[GLOBALS]] = !{[[CNST:![0-9]*]]}

// CHECK: [[CNST]] = !MDGlobalVariable(name: "cnst",
// CHECK-SAME:                         scope: [[NS:![0-9]*]]
// CHECK: [[NS]] = !MDNamespace(name: "ns"

