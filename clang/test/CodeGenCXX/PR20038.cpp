// RUN: %clang_cc1 -g -emit-llvm %s -o - | FileCheck %s

struct C {
  ~C();
};
extern bool b;
// CHECK: call {{.*}}, !dbg [[DTOR_CALL_LOC:![0-9]*]]
// CHECK: [[FUN4:.*]] = {{.*}}; [ DW_TAG_subprogram ] {{.*}} [def] [fun4]
// CHECK: [[DTOR_CALL_LOC]] = metadata !{i32 [[@LINE+2]], i32 0, metadata [[FUN4_BLOCK:.*]], null}
// CHECK: [[FUN4_BLOCK]] = metadata !{{{[^,]*}}, {{[^,]*}}, metadata [[FUN4]],
void fun4() { b && (C(), 1); }
