; RUN: opt < %s -loop-vectorize -pass-remarks-analysis='loop-vectorize' -mtriple=x86_64-unknown-linux -S 2>&1 | FileCheck %s

; Verify analysis remarks are generated when interleaving is not beneficial.
; CHECK: remark: vectorization-remarks-profitable.c:4:14: the cost-model indicates that vectorization is not beneficial
; CHECK: remark: vectorization-remarks-profitable.c:4:14: the cost-model indicates that interleaving is not beneficial and is explicitly disabled or interleave count is set to 1
; CHECK: remark: vectorization-remarks-profitable.c:11:14: the cost-model indicates that vectorization is not beneficial
; CHECK: remark: vectorization-remarks-profitable.c:11:14: the cost-model indicates that interleaving is not beneficial

; First loop.
;  #pragma clang loop interleave(disable) unroll(disable)
;  for(int i = 0; i < n; i++) {
;    out[i] = in[i];
;  }

; Second loop.
;  #pragma clang loop unroll(disable)
;  for(int i = 0; i < n; i++) {
;    out[i] = in[i];
;  }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; Function Attrs: nounwind ssp uwtable
define void @do_not_interleave(float* nocapture %out, float* nocapture readonly %in, i32 %n) #0 {
entry:
  %cmp.7 = icmp sgt i32 %n, 0, !dbg !3
  br i1 %cmp.7, label %for.body.preheader, label %for.cond.cleanup, !dbg !8

for.body.preheader:                               ; preds = %entry
  br label %for.body, !dbg !9

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup, !dbg !10

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void, !dbg !10

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds float, float* %in, i64 %indvars.iv, !dbg !9
  %0 = bitcast float* %arrayidx to i32*, !dbg !9
  %1 = load i32, i32* %0, align 4, !dbg !9, !tbaa !11
  %arrayidx2 = getelementptr inbounds float, float* %out, i64 %indvars.iv, !dbg !15
  %2 = bitcast float* %arrayidx2 to i32*, !dbg !16
  store i32 %1, i32* %2, align 4, !dbg !16, !tbaa !11
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !8
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !8
  %exitcond = icmp eq i32 %lftr.wideiv, %n, !dbg !8
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body, !dbg !8, !llvm.loop !17
}

; Function Attrs: nounwind ssp uwtable
define void @interleave_not_profitable(float* nocapture %out, float* nocapture readonly %in, i32 %n) #0 {
entry:
  %cmp.7 = icmp sgt i32 %n, 0, !dbg !20
  br i1 %cmp.7, label %for.body.preheader, label %for.cond.cleanup, !dbg !22

for.body.preheader:                               ; preds = %entry
  br label %for.body, !dbg !23

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup, !dbg !24

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void, !dbg !24

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds float, float* %in, i64 %indvars.iv, !dbg !23
  %0 = bitcast float* %arrayidx to i32*, !dbg !23
  %1 = load i32, i32* %0, align 4, !dbg !23, !tbaa !11
  %arrayidx2 = getelementptr inbounds float, float* %out, i64 %indvars.iv, !dbg !25
  %2 = bitcast float* %arrayidx2 to i32*, !dbg !26
  store i32 %1, i32* %2, align 4, !dbg !26, !tbaa !11
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !22
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !22
  %exitcond = icmp eq i32 %lftr.wideiv, %n, !dbg !22
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body, !dbg !22, !llvm.loop !27
}

attributes #0 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"PIC Level", i32 2}
!2 = !{!"clang version 3.7.0"}
!3 = !DILocation(line: 3, column: 20, scope: !4)
!4 = !DISubprogram(name: "do_not_interleave", scope: !5, file: !5, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, function: void (float*, float*, i32)* @do_not_interleave, variables: !7)
!5 = !DIFile(filename: "vectorization-remarks-profitable.c", directory: "")
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 3, column: 3, scope: !4)
!9 = !DILocation(line: 4, column: 14, scope: !4)
!10 = !DILocation(line: 6, column: 1, scope: !4)
!11 = !{!12, !12, i64 0}
!12 = !{!"float", !13, i64 0}
!13 = !{!"omnipotent char", !14, i64 0}
!14 = !{!"Simple C/C++ TBAA"}
!15 = !DILocation(line: 4, column: 5, scope: !4)
!16 = !DILocation(line: 4, column: 12, scope: !4)
!17 = distinct !{!17, !18, !19}
!18 = !{!"llvm.loop.interleave.count", i32 1}
!19 = !{!"llvm.loop.unroll.disable"}
!20 = !DILocation(line: 10, column: 20, scope: !21)
!21 = !DISubprogram(name: "interleave_not_profitable", scope: !5, file: !5, line: 8, type: !6, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: true, function: void (float*, float*, i32)* @interleave_not_profitable, variables: !7)
!22 = !DILocation(line: 10, column: 3, scope: !21)
!23 = !DILocation(line: 11, column: 14, scope: !21)
!24 = !DILocation(line: 13, column: 1, scope: !21)
!25 = !DILocation(line: 11, column: 5, scope: !21)
!26 = !DILocation(line: 11, column: 12, scope: !21)
!27 = distinct !{!27, !19}
