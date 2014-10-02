// RUN: %clang_cc1 -triple %itanium_abi_triple -g -emit-llvm  %s -o - | FileCheck %s

struct C {
  ~C();
};
extern bool b;
// CHECK: call {{.*}}, !dbg [[DTOR_CALL1_LOC:![0-9]*]]
// CHECK: call {{.*}}, !dbg [[DTOR_CALL2_LOC:![0-9]*]]
// CHECK: [[FUN1:.*]] = {{.*}}; [ DW_TAG_subprogram ] {{.*}} [def] [fun1]
// CHECK: [[FUN2:.*]] = {{.*}}; [ DW_TAG_subprogram ] {{.*}} [def] [fun2]
// CHECK: [[DTOR_CALL1_LOC]] = metadata !{i32 [[@LINE+2]], i32 0, metadata [[FUN1_BLOCK:.*]], null}
// CHECK: [[FUN1_BLOCK]] = metadata !{metadata !"0xb{{[^,]*}}", {{[^,]*}}, metadata [[FUN1]]}
void fun1() { b && (C(), 1); }
// CHECK: [[DTOR_CALL2_LOC]] = metadata !{i32 [[@LINE+2]], i32 0, metadata [[FUN2_BLOCK1:.*]], null}
// CHECK: [[FUN2_BLOCK1]] = metadata !{metadata !"0xb{{[^,]*}}", {{[^,]*}}, metadata [[FUN2]]}
bool fun2() { return (C(), b) && 0; }
