; RUN: opt < %s -argpromotion -S | FileCheck %s
; CHECK: call void @test(), !dbg !1
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"
define internal void @test(i32* %X) {
  ret void
}

define void @caller() {
  call void @test(i32* null), !dbg !1
  ret void
}

!llvm.module.flags = !{!3}

!1 = metadata !{i32 8, i32 0, metadata !2, null}
!2 = metadata !{}
!3 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
