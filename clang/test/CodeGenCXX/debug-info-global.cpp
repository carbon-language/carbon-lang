// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -emit-llvm -g %s -o - | FileCheck %s

// Multiple references to the same constant should result in only one entry in
// the globals list.

namespace ns {
const int cnst = 42;
}
int f1() {
  return ns::cnst + ns::cnst;
}

// CHECK: !"0x11\00{{.*}}"{{.*}}, [[GLOBALS:![0-9]*]], {{![0-9]*}}} ; [ DW_TAG_compile_unit ]

// CHECK: [[GLOBALS]] = !{[[CNST:![0-9]*]]}

// CHECK: [[CNST]] = !{!"0x34\00cnst\00{{.*}}", [[NS:![0-9]*]], {{[^,]+, [^,]+, [^,]+, [^,]+}}} ; [ DW_TAG_variable ] [cnst]
// CHECK: [[NS]] = {{.*}}; [ DW_TAG_namespace ] [ns]

