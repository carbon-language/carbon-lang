; RUN: opt < %s -loop-vectorize -S 2>&1 | FileCheck %s

; Verify warning is generated when vectorization/ interleaving is explicitly specified and fails to occur.
; CHECK: warning: no_array_bounds.cpp:5:5: loop not vectorized: failed explicitly specified loop vectorization
; CHECK: warning: no_array_bounds.cpp:10:5: loop not interleaved: failed explicitly specified loop interleaving

;  #pragma clang loop vectorize(enable)
;  for (int i = 0; i < number; i++) {
;    A[B[i]]++;
;  }

;  #pragma clang loop vectorize(disable) interleave(enable)
;  for (int i = 0; i < number; i++) {
;    B[A[i]]++;
;  }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind ssp uwtable
define void @_Z4testPiS_i(i32* nocapture %A, i32* nocapture %B, i32 %number) #0 {
entry:
  %cmp25 = icmp sgt i32 %number, 0, !dbg !10
  br i1 %cmp25, label %for.body.preheader, label %for.end15, !dbg !10, !llvm.loop !12

for.body.preheader:                               ; preds = %entry
  br label %for.body, !dbg !14

for.cond5.preheader:                              ; preds = %for.body
  br i1 %cmp25, label %for.body7.preheader, label %for.end15, !dbg !16, !llvm.loop !18

for.body7.preheader:                              ; preds = %for.cond5.preheader
  br label %for.body7, !dbg !20

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv27 = phi i64 [ %indvars.iv.next28, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32* %B, i64 %indvars.iv27, !dbg !14
  %0 = load i32* %arrayidx, align 4, !dbg !14, !tbaa !22
  %idxprom1 = sext i32 %0 to i64, !dbg !14
  %arrayidx2 = getelementptr inbounds i32* %A, i64 %idxprom1, !dbg !14
  %1 = load i32* %arrayidx2, align 4, !dbg !14, !tbaa !22
  %inc = add nsw i32 %1, 1, !dbg !14
  store i32 %inc, i32* %arrayidx2, align 4, !dbg !14, !tbaa !22
  %indvars.iv.next28 = add nuw nsw i64 %indvars.iv27, 1, !dbg !10
  %lftr.wideiv29 = trunc i64 %indvars.iv.next28 to i32, !dbg !10
  %exitcond30 = icmp eq i32 %lftr.wideiv29, %number, !dbg !10
  br i1 %exitcond30, label %for.cond5.preheader, label %for.body, !dbg !10, !llvm.loop !12

for.body7:                                        ; preds = %for.body7.preheader, %for.body7
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body7 ], [ 0, %for.body7.preheader ]
  %arrayidx9 = getelementptr inbounds i32* %A, i64 %indvars.iv, !dbg !20
  %2 = load i32* %arrayidx9, align 4, !dbg !20, !tbaa !22
  %idxprom10 = sext i32 %2 to i64, !dbg !20
  %arrayidx11 = getelementptr inbounds i32* %B, i64 %idxprom10, !dbg !20
  %3 = load i32* %arrayidx11, align 4, !dbg !20, !tbaa !22
  %inc12 = add nsw i32 %3, 1, !dbg !20
  store i32 %inc12, i32* %arrayidx11, align 4, !dbg !20, !tbaa !22
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !16
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !16
  %exitcond = icmp eq i32 %lftr.wideiv, %number, !dbg !16
  br i1 %exitcond, label %for.end15.loopexit, label %for.body7, !dbg !16, !llvm.loop !18

for.end15.loopexit:                               ; preds = %for.body7
  br label %for.end15

for.end15:                                        ; preds = %for.end15.loopexit, %entry, %for.cond5.preheader
  ret void, !dbg !26
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 2}
!1 = metadata !{metadata !"no_array_bounds.cpp", metadata !"."}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"test", metadata !"test", metadata !"", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, void (i32*, i32*, i32)* @_Z4testPiS_i, null, null, metadata !2, i32 2}
!5 = metadata !{i32 786473, metadata !1}
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !2, i32 0, null, null, null}
!7 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!8 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!9 = metadata !{metadata !"clang version 3.5.0"}
!10 = metadata !{i32 4, i32 8, metadata !11, null}
!11 = metadata !{i32 786443, metadata !1, metadata !4, i32 4, i32 3, i32 0, i32 0}
!12 = metadata !{metadata !12, metadata !13}
!13 = metadata !{metadata !"llvm.loop.vectorize.enable", i1 true}
!14 = metadata !{i32 5, i32 5, metadata !15, null}
!15 = metadata !{i32 786443, metadata !1, metadata !11, i32 4, i32 36, i32 0, i32 1}
!16 = metadata !{i32 9, i32 8, metadata !17, null}
!17 = metadata !{i32 786443, metadata !1, metadata !4, i32 9, i32 3, i32 0, i32 2}
!18 = metadata !{metadata !18, metadata !13, metadata !19}
!19 = metadata !{metadata !"llvm.loop.vectorize.width", i32 1}
!20 = metadata !{i32 10, i32 5, metadata !21, null}
!21 = metadata !{i32 786443, metadata !1, metadata !17, i32 9, i32 36, i32 0, i32 3}
!22 = metadata !{metadata !23, metadata !23, i64 0}
!23 = metadata !{metadata !"int", metadata !24, i64 0}
!24 = metadata !{metadata !"omnipotent char", metadata !25, i64 0}
!25 = metadata !{metadata !"Simple C/C++ TBAA"}
!26 = metadata !{i32 12, i32 1, metadata !4, null}
