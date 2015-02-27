; RUN: llc -mcpu=corei7 -no-stack-coloring=false < %s

; Make sure that we don't crash when dbg values are used.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define void @foo() nounwind uwtable ssp {
entry:
  %x.i = alloca i8, align 1
  %y.i = alloca [256 x i8], align 16
  %0 = getelementptr inbounds [256 x i8], [256 x i8]* %y.i, i64 0, i64 0
  br label %for.body

for.body:
  call void @llvm.lifetime.end(i64 -1, i8* %0) nounwind
  call void @llvm.lifetime.start(i64 -1, i8* %x.i) nounwind
  call void @llvm.dbg.declare(metadata i8* %x.i, metadata !22, metadata !{!"0x102"}) nounwind
  br label %for.body
}

declare void @llvm.lifetime.start(i64, i8* nocapture) nounwind

declare void @llvm.lifetime.end(i64, i8* nocapture) nounwind

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!23}
!0 = !{!"0x11\001\00clang\001\00\000\00\000", !1, !2, !2, null, null, null} ; [ DW_TAG_compile_unit ]
!1 = !{!"t.c", !""}
!16 = !{!"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ]
!2 = !{i32 0}
!22 = !{!"0x100\00x\0016\000", null, !2, !16} ; [ DW_TAG_auto_variable ]
!23 = !{i32 1, !"Debug Info Version", i32 2}
