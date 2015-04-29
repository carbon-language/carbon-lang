// RUN: %clang_cc1 -emit-llvm -gdwarf-4 -triple x86_64-linux-gnu %s -o - | FileCheck %s

// Make sure that we emit a global variable for each of the members of the
// anonymous union.

static union {
  int c;
  int d;
  union {
    int a;
  };
  struct {
    int b;
  };
};

int test_it() {
  c = 1;
  d = 2;
  a = 4;
  return (c == 1);
}

// CHECK: [[FILE:.*]] = !DIFile(filename: "{{.*}}debug-info-anon-union-vars.cpp",
// CHECK: !DIGlobalVariable(name: "c",{{.*}} file: [[FILE]], line: 6,{{.*}} isLocal: true, isDefinition: true
// CHECK: !DIGlobalVariable(name: "d",{{.*}} file: [[FILE]], line: 6,{{.*}} isLocal: true, isDefinition: true
// CHECK: !DIGlobalVariable(name: "a",{{.*}} file: [[FILE]], line: 6,{{.*}} isLocal: true, isDefinition: true
// CHECK: !DIGlobalVariable(name: "b",{{.*}} file: [[FILE]], line: 6,{{.*}} isLocal: true, isDefinition: true
