; RUN: opt < %s -loop-vectorize -force-vector-width=4 -S -pass-remarks-missed='loop-vectorize' -pass-remarks-analysis='loop-vectorize' 2>&1 | FileCheck %s

; CHECK: remark: source.cpp:4:5: loop not vectorized: loop contains a switch statement
; CHECK: remark: source.cpp:4:5: loop not vectorized: vectorization is explicitly enabled with width 4
; CHECK: warning: source.cpp:4:5: loop not vectorized: failed explicitly specified loop vectorization

; CHECK: _Z11test_switchPii
; CHECK-NOT: x i32>
; CHECK: ret

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind optsize ssp uwtable
define void @_Z11test_switchPii(i32* nocapture %A, i32 %Length) #0 {
entry:
  %cmp18 = icmp sgt i32 %Length, 0, !dbg !10
  br i1 %cmp18, label %for.body.preheader, label %for.end, !dbg !10, !llvm.loop !12

for.body.preheader:                               ; preds = %entry
  br label %for.body, !dbg !14

for.body:                                         ; preds = %for.body.preheader, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32* %A, i64 %indvars.iv, !dbg !14
  %0 = load i32* %arrayidx, align 4, !dbg !14, !tbaa !16
  switch i32 %0, label %for.inc [
    i32 0, label %sw.bb
    i32 1, label %sw.bb3
  ], !dbg !14

sw.bb:                                            ; preds = %for.body
  %1 = trunc i64 %indvars.iv to i32, !dbg !20
  %mul = shl nsw i32 %1, 1, !dbg !20
  br label %for.inc, !dbg !22

sw.bb3:                                           ; preds = %for.body
  %2 = trunc i64 %indvars.iv to i32, !dbg !23
  store i32 %2, i32* %arrayidx, align 4, !dbg !23, !tbaa !16
  br label %for.inc, !dbg !23

for.inc:                                          ; preds = %sw.bb3, %for.body, %sw.bb
  %storemerge = phi i32 [ %mul, %sw.bb ], [ 0, %for.body ], [ 0, %sw.bb3 ]
  store i32 %storemerge, i32* %arrayidx, align 4, !dbg !20, !tbaa !16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !10
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !10
  %exitcond = icmp eq i32 %lftr.wideiv, %Length, !dbg !10
  br i1 %exitcond, label %for.end.loopexit, label %for.body, !dbg !10, !llvm.loop !12

for.end.loopexit:                                 ; preds = %for.inc
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void, !dbg !24
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 2}
!1 = metadata !{metadata !"source.cpp", metadata !"."}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"test_switch", metadata !"test_switch", metadata !"", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, void (i32*, i32)* @_Z11test_switchPii, null, null, metadata !2, i32 1}
!5 = metadata !{i32 786473, metadata !1}
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !2, i32 0, null, null, null}
!7 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!8 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!9 = metadata !{metadata !"clang version 3.5.0"}
!10 = metadata !{i32 3, i32 8, metadata !11, null}
!11 = metadata !{i32 786443, metadata !1, metadata !4, i32 3, i32 3, i32 0, i32 0}
!12 = metadata !{metadata !12, metadata !13, metadata !13}
!13 = metadata !{metadata !"llvm.loop.vectorize.enable", i1 true}
!14 = metadata !{i32 4, i32 5, metadata !15, null}
!15 = metadata !{i32 786443, metadata !1, metadata !11, i32 3, i32 36, i32 0, i32 1}
!16 = metadata !{metadata !17, metadata !17, i64 0}
!17 = metadata !{metadata !"int", metadata !18, i64 0}
!18 = metadata !{metadata !"omnipotent char", metadata !19, i64 0}
!19 = metadata !{metadata !"Simple C/C++ TBAA"}
!20 = metadata !{i32 6, i32 7, metadata !21, null}
!21 = metadata !{i32 786443, metadata !1, metadata !15, i32 4, i32 18, i32 0, i32 2}
!22 = metadata !{i32 7, i32 5, metadata !21, null}
!23 = metadata !{i32 9, i32 7, metadata !21, null}
!24 = metadata !{i32 14, i32 1, metadata !4, null}
