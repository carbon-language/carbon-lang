// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin10 %s -o - | FileCheck %s

// Check the line numbers for cleanup code with EH in combinatin with
// simple return expressions.

// CHECK: define {{.*}}foo
// CHECK: call void @_ZN1CD1Ev(%class.C* {{.*}}), !dbg ![[CLEANUP:[0-9]+]]
// CHECK: ret i32 0, !dbg ![[RET:[0-9]+]]

class C {
public:
  ~C() {}
  int i;
};

int foo()
{
  C c;
  c.i = 42;
  // This breakpoint should be at/before the cleanup code.
  // CHECK: ![[CLEANUP]] = metadata !{i32 [[@LINE+1]], i32 0, metadata !{{.*}}, null}
  return 0;
  // CHECK: ![[RET]] = metadata !{i32 [[@LINE+1]], i32 0, metadata !{{.*}}, null}
}
