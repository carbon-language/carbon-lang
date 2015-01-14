// RUN: %clang_cc1 -triple %itanium_abi_triple -g -mllvm -no-discriminators -emit-llvm  %s -o - | FileCheck %s

struct C {
  ~C();
};
extern bool b;
// CHECK: call {{.*}}, !dbg [[DTOR_CALL1_LOC:![0-9]*]]
// CHECK: call {{.*}}, !dbg [[DTOR_CALL2_LOC:![0-9]*]]
// CHECK: [[FUN1:.*]] = {{.*}}; [ DW_TAG_subprogram ] {{.*}} [def] [fun1]
// CHECK: [[FUN2:.*]] = {{.*}}; [ DW_TAG_subprogram ] {{.*}} [def] [fun2]
// CHECK: [[DTOR_CALL1_LOC]] = !{i32 [[@LINE+1]], i32 0, [[FUN1]], null}
void fun1() { b && (C(), 1); }
// CHECK: [[DTOR_CALL2_LOC]] = !{i32 [[@LINE+1]], i32 0, [[FUN2]], null}
bool fun2() { return (C(), b) && 0; }
