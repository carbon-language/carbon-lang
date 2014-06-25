; RUN: opt < %s -loop-vectorize -force-vector-width=4 -S -pass-remarks-missed='loop-vectorize' -pass-remarks-analysis='loop-vectorize' 2>&1 | FileCheck %s

; C/C++ code for control flow test
; int test(int *A, int Length) {
;   for (int i = 0; i < Length; i++) {
;     if (A[i] > 10.0) goto end;
;     A[i] = 0;
;   }
; end:
;   return 0;
; }

; CHECK: remark: source.cpp:5:9: loop not vectorized: loop control flow is not understood by vectorizer
; CHECK: remark: source.cpp:5:9: loop not vectorized: vectorization was not specified

; CHECK: _Z4testPii
; CHECK-NOT: x i32>
; CHECK: ret

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind optsize ssp uwtable
define i32 @_Z4testPii(i32* nocapture %A, i32 %Length) #0 {
entry:
  %cmp8 = icmp sgt i32 %Length, 0, !dbg !10
  br i1 %cmp8, label %for.body.preheader, label %end, !dbg !10

for.body.preheader:                               ; preds = %entry
  br label %for.body, !dbg !12

for.body:                                         ; preds = %for.body.preheader, %if.else
  %indvars.iv = phi i64 [ %indvars.iv.next, %if.else ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32* %A, i64 %indvars.iv, !dbg !12
  %0 = load i32* %arrayidx, align 4, !dbg !12, !tbaa !15
  %cmp1 = icmp sgt i32 %0, 10, !dbg !12
  br i1 %cmp1, label %end.loopexit, label %if.else, !dbg !12

if.else:                                          ; preds = %for.body
  store i32 0, i32* %arrayidx, align 4, !dbg !19, !tbaa !15
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !10
  %1 = trunc i64 %indvars.iv.next to i32, !dbg !10
  %cmp = icmp slt i32 %1, %Length, !dbg !10
  br i1 %cmp, label %for.body, label %end.loopexit, !dbg !10

end.loopexit:                                     ; preds = %if.else, %for.body
  br label %end

end:                                              ; preds = %end.loopexit, %entry
  ret i32 0, !dbg !20
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 2}
!1 = metadata !{metadata !"source.cpp", metadata !"."}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"test", metadata !"test", metadata !"", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (i32*, i32)* @_Z4testPii, null, null, metadata !2, i32 2}
!5 = metadata !{i32 786473, metadata !1}
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !2, i32 0, null, null, null}
!7 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!8 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!9 = metadata !{metadata !"clang version 3.5.0"}
!10 = metadata !{i32 3, i32 8, metadata !11, null}
!11 = metadata !{i32 786443, metadata !1, metadata !4, i32 3, i32 3, i32 0, i32 0}
!12 = metadata !{i32 5, i32 9, metadata !13, null}
!13 = metadata !{i32 786443, metadata !1, metadata !14, i32 5, i32 9, i32 0, i32 2}
!14 = metadata !{i32 786443, metadata !1, metadata !11, i32 4, i32 3, i32 0, i32 1}
!15 = metadata !{metadata !16, metadata !16, i64 0}
!16 = metadata !{metadata !"int", metadata !17, i64 0}
!17 = metadata !{metadata !"omnipotent char", metadata !18, i64 0}
!18 = metadata !{metadata !"Simple C/C++ TBAA"}
!19 = metadata !{i32 8, i32 7, metadata !13, null}
!20 = metadata !{i32 12, i32 3, metadata !4, null}
