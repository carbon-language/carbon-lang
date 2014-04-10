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

// CHECK: [[FILE:.*]] = {{.*}}[ DW_TAG_file_type ] [{{.*}}debug-info-anon-union-vars.cpp]
// CHECK: [[FILE]]{{.*}}[ DW_TAG_variable ] [c] [line 6] [local] [def]
// CHECK: [[FILE]]{{.*}}[ DW_TAG_variable ] [d] [line 6] [local] [def]
// CHECK: [[FILE]]{{.*}}[ DW_TAG_variable ] [a] [line 6] [local] [def]
// CHECK: [[FILE]]{{.*}}[ DW_TAG_variable ] [b] [line 6] [local] [def]
