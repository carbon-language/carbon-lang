// RUN: %clang_cc1 %s -emit-llvm -o - -fblocks -triple x86_64-apple-darwin10 | FileCheck %s
// rdar://8893785

void MYFUNC() {
// CHECK: [[T1:%.*]] = bitcast i8* ()*
// CHECK-NEXT: [[FORWARDING:%.*]] = getelementptr inbounds [[N_T:%.*]]* [[N:%.*]], i32 0, i32 1
// CHECK-NEXT: [[T0:%.*]] = load [[N_T]]** [[FORWARDING]]
// CHECK-NEXT: [[OBSERVER:%.*]] = getelementptr inbounds [[N_T]]* [[T0]], i32 0, i32 6
// CHECK-NEXT: store i8* [[T1]], i8** [[OBSERVER]]
  __block id observer = ^{ return observer; };
}

