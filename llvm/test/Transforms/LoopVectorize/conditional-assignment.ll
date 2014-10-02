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

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.6.0", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 2}
!1 = metadata !{metadata !"source.c", metadata !"."}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"conditional_store", metadata !"conditional_store", metadata !"", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, void (i32*)* @conditional_store, null, null, metadata !2, i32 1}
!5 = metadata !{i32 786473, metadata !1}
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !2, i32 0, null, null, null}
!7 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!8 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!9 = metadata !{metadata !"clang version 3.6.0"}
!10 = metadata !{i32 2, i32 8, metadata !11, null}
!11 = metadata !{i32 786443, metadata !1, metadata !4, i32 2, i32 3, i32 0, i32 0}
!12 = metadata !{i32 3, i32 9, metadata !13, null}
!13 = metadata !{i32 786443, metadata !1, metadata !11, i32 3, i32 9, i32 0, i32 1}
!14 = metadata !{metadata !15, metadata !15, i64 0}
!15 = metadata !{metadata !"int", metadata !16, i64 0}
!16 = metadata !{metadata !"omnipotent char", metadata !17, i64 0}
!17 = metadata !{metadata !"Simple C/C++ TBAA"}
!18 = metadata !{i32 3, i32 29, metadata !13, null}
!19 = metadata !{i32 4, i32 1, metadata !4, null}
