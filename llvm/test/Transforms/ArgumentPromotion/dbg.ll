; RUN: opt < %s -argpromotion -S | FileCheck %s
; CHECK: call void @test(), !dbg !1

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
