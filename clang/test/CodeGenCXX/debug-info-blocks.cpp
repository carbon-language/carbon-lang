// RUN: %clang_cc1 %s -gline-tables-only -fblocks -S -emit-llvm -o - | FileCheck %s

struct A {
  A();
  A(const A &);
  ~A();
};

void test() {
  __block A a;
}

// CHECK: !MDSubprogram(name: "__Block_byref_object_copy_",
// CHECK-SAME:          line: 10,
// CHECK-SAME:          isLocal: true, isDefinition: true
// CHECK: !MDSubprogram(name: "__Block_byref_object_dispose_",
// CHECK-SAME:          line: 10,
// CHECK-SAME:          isLocal: true, isDefinition: true
