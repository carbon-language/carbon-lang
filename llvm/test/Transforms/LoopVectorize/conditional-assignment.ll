; RUN: opt < %s -loop-vectorize -S -pass-remarks-missed='loop-vectorize' -pass-remarks-analysis='loop-vectorize' 2>&1 | FileCheck %s

; CHECK: remark: source.c:2:8: loop not vectorized: store that is conditionally executed prevents vectorization

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; Function Attrs: nounwind ssp uwtable
define void @conditional_store(i32* noalias nocapture %indices) #0 {
entry:
  br label %for.body, !dbg !10

for.body:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.inc ]
  %arrayidx = getelementptr inbounds i32* %indices, i64 %indvars.iv, !dbg !12
  %0 = load i32* %arrayidx, align 4, !dbg !12, !tbaa !14
  %cmp1 = icmp eq i32 %0, 1024, !dbg !12
  br i1 %cmp1, label %if.then, label %for.inc, !dbg !12

if.then:                                          ; preds = %for.body
  store i32 0, i32* %arrayidx, align 4, !dbg !18, !tbaa !14
  br label %for.inc, !dbg !18

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !10
  %exitcond = icmp eq i64 %indvars.iv.next, 4096, !dbg !10
  br i1 %exitcond, label %for.end, label %for.body, !dbg !10

for.end:                                          ; preds = %for.inc
  ret void, !dbg !19
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.6.0\001\00\000\00\002", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !"source.c", metadata !"."}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00conditional_store\00conditional_store\00\001\000\001\000\006\00256\001\001", metadata !1, metadata !5, metadata !6, null, void (i32*)* @conditional_store, null, null, metadata !2} ; [ DW_TAG_subprogram ]
!5 = metadata !{metadata !"0x29", metadata !1} ; [ DW_TAG_file_type ]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !2, null, null, null} ; [ DW_TAG_subroutine_type ]
!7 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!8 = metadata !{i32 2, metadata !"Debug Info Version", i32 2}
!9 = metadata !{metadata !"clang version 3.6.0"}
!10 = metadata !{i32 2, i32 8, metadata !11, null}
!11 = metadata !{metadata !"0xb\002\003\000", metadata !1, metadata !4} ; [ DW_TAG_lexical_block ]
!12 = metadata !{i32 3, i32 9, metadata !13, null}
!13 = metadata !{metadata !"0xb\003\009\000", metadata !1, metadata !11} ; [ DW_TAG_lexical_block ]
!14 = metadata !{metadata !15, metadata !15, i64 0}
!15 = metadata !{metadata !"int", metadata !16, i64 0}
!16 = metadata !{metadata !"omnipotent char", metadata !17, i64 0}
!17 = metadata !{metadata !"Simple C/C++ TBAA"}
!18 = metadata !{i32 3, i32 29, metadata !13, null}
!19 = metadata !{i32 4, i32 1, metadata !4, null}
