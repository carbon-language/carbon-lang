; RUN: opt -globalopt %s -S -o - | FileCheck %s
; PR10047


%0 = type { i32, void ()* }
%struct.A = type { [100 x i32] }

; CHECK: @a
@a = global %struct.A zeroinitializer, align 4
@llvm.global_ctors = appending global [1 x %0] [%0 { i32 65535, void ()* @_GLOBAL__I_a }]

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind

; CHECK-NOT: GLOBAL__I_a
define internal void @_GLOBAL__I_a() nounwind {
entry:
  tail call void @llvm.memset.p0i8.i64(i8* bitcast (%struct.A* @a to i8*), i8 0, i64 400, i32 4, i1 false) nounwind
  ret void
}
