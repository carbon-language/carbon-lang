// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -g %s -o - | FileCheck %s

// CHECK: !"0x11\00{{.*}}", {{[^,]*}}, [[ENUMS:![0-9]*]], {{.*}}} ; [ DW_TAG_compile_unit ]
// CHECK: [[ENUMS]] = !{[[E1:![0-9]*]], [[E2:![0-9]*]], [[E3:![0-9]*]]}

namespace test1 {
// CHECK: [[E1]] = !{!"0x4\00{{.*}}", {{[^,]*}}, [[TEST1:![0-9]*]], {{.*}}, [[TEST1_ENUMS:![0-9]*]], null, null, !"_ZTSN5test11eE"} ; [ DW_TAG_enumeration_type ] [e]
// CHECK: [[TEST1]] = {{.*}} ; [ DW_TAG_namespace ] [test1]
// CHECK: [[TEST1_ENUMS]] = !{[[TEST1_E:![0-9]*]]}
// CHECK: [[TEST1_E]] = !{!"0x28\00E\000"} ; [ DW_TAG_enumerator ] [E :: 0]
enum e { E };
void foo() {
  int v = E;
}
}

namespace test2 {
// rdar://8195980
// CHECK: [[E2]] = !{!"0x4\00{{.*}}", {{[^,]*}}, [[TEST2:![0-9]*]], {{.*}}, [[TEST1_ENUMS]], null, null, !"_ZTSN5test21eE"} ; [ DW_TAG_enumeration_type ] [e]
// CHECK: [[TEST2]] = {{.*}} ; [ DW_TAG_namespace ] [test2]
enum e { E };
bool func(int i) {
  return i == E;
}
}

namespace test3 {
// CHECK: [[E3]] = !{!"0x4\00{{.*}}", {{[^,]*}}, [[TEST3:![0-9]*]], {{.*}}, [[TEST3_ENUMS:![0-9]*]], null, null, !"_ZTSN5test31eE"} ; [ DW_TAG_enumeration_type ] [e]
// CHECK: [[TEST3]] = {{.*}} ; [ DW_TAG_namespace ] [test3]
// CHECK: [[TEST3_ENUMS]] = !{[[TEST3_E:![0-9]*]]}
// CHECK: [[TEST3_E]] = !{!"0x28\00E\00-1"} ; [ DW_TAG_enumerator ] [E :: -1]
enum e { E = -1 };
void func() {
  e x;
}
}

namespace test4 {
// Don't try to build debug info for a dependent enum.
// CHECK-NOT: test4
template <typename T>
struct S {
  enum e { E = T::v };
};
}
