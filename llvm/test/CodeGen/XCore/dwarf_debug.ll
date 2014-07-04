; RUN: llc < %s -mtriple=xcore-unknown-unknown -O0 | FileCheck %s

; target datalayout = "e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i64:32-f64:32-a:0:32-n32"
; target triple = "xcore"

; CHECK-LABEL: f
; CHECK: entsp 2
; ...the prologue...
; CHECK: .loc 1 2 0 prologue_end      # :2:0
; CHECK: add r0, r0, 1
; CHECK: retsp 2
define i32 @f(i32 %a) {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %a.addr}, metadata !11), !dbg !12
  %0 = load i32* %a.addr, align 4, !dbg !12
  %add = add nsw i32 %0, 1, !dbg !12
  ret i32 %add, !dbg !12
}

declare void @llvm.dbg.declare(metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 1}
!1 = metadata !{metadata !"", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"f", metadata !"f", metadata !"", i32 2, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32)* @f, null, null, metadata !2, i32 2}
!5 = metadata !{i32 786473, metadata !1}
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null}
!7 = metadata !{metadata !8, metadata !8}
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5}
!9 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!10 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!11 = metadata !{i32 786689, metadata !4, metadata !"a", metadata !5, i32 16777218, metadata !8, i32 0, i32 0}
!12 = metadata !{i32 2, i32 0, metadata !4, null}

