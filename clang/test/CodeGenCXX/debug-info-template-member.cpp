// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s

struct MyClass {
  template <int i> int add(int j) {
    return i + j;
  }
};

int add2(int x) {
  return MyClass().add<2>(x);
}

inline int add3(int x) {
  return MyClass().add<3>(x); // even though add<3> is ODR used, don't emit it since we don't codegen it
}

// CHECK: metadata [[C_MEM:![0-9]*]], i32 0, null, null} ; [ DW_TAG_structure_type ] [MyClass]
// CHECK: [[C_MEM]] = metadata !{metadata [[C_TEMP:![0-9]*]]}
// CHECK: [[C_TEMP]] = {{.*}} ; [ DW_TAG_subprogram ] [line 4] [add<2>]
