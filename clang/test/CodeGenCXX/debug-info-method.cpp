// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -std=c++11 -g %s -o - | FileCheck %s
// CHECK: !MDCompositeType(tag: DW_TAG_class_type, name: "A",{{.*}} identifier: "_ZTS1A")
// CHECK: !MDSubprogram(name: "foo", linkageName: "_ZN1A3fooEiS_3$_0"
// CHECK-SAME:          DIFlagProtected
// CHECK: ![[THISTYPE:[0-9]+]] = !MDDerivedType(tag: DW_TAG_pointer_type, baseType: !"_ZTS1A"
// CHECK-SAME:                                  DIFlagArtificial
// CHECK: !MDDerivedType(tag: DW_TAG_ptr_to_member_type
// CHECK: !MDDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: ![[MEMFUNTYPE:[0-9]+]]
// CHECK: ![[MEMFUNTYPE]] = !MDSubroutineType(types: ![[MEMFUNARGS:[0-9]+]])
// CHECK: ![[MEMFUNARGS]] = {{.*}}, ![[THISTYPE]],
// CHECK: !MDLocalVariable(tag: DW_TAG_arg_variable
// CHECK: !MDLocalVariable(tag: DW_TAG_arg_variable
// CHECK: !MDLocalVariable(tag: DW_TAG_arg_variable
union {
  int a;
  float b;
} u;

class A {
protected:
  void foo(int, A, decltype(u));
}; 

void A::foo(int, A, decltype(u)) {
}

A a;

int A::*x = 0;
int (A::*y)(int) = 0;
