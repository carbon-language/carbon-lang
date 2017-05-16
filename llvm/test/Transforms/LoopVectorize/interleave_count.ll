; RUN: opt < %s -loop-vectorize -force-target-instruction-cost=0 -force-vector-width=2 -force-vector-interleave=1 -instcombine -S | FileCheck %s

; CHECK-LABEL: @copy(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i64, i64* %a, i64 [[INDEX]]
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr inbounds i64, i64* %b, i64 [[INDEX]]
; CHECK-NEXT:    [[TMP4:%.*]] = bitcast i64* [[TMP3]] to <2 x i64>*
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <2 x i64>, <2 x i64>* [[TMP4]], align 8
; CHECK-NEXT:    [[TMP5:%.*]] = bitcast i64* [[TMP2]] to <2 x i64>*
; CHECK-NEXT:    store <2 x i64> [[WIDE_LOAD]], <2 x i64>* [[TMP5]], align 8
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], 2
; CHECK:         br i1 {{.*}}, label %middle.block, label %vector.body
;
define void @copy(i64* %a, i64* %b, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %tmp0 = getelementptr inbounds i64, i64* %a, i64 %i
  %tmp1 = getelementptr inbounds i64, i64* %b, i64 %i
  %tmp3 = load i64, i64* %tmp1, align 8
  store i64 %tmp3, i64* %tmp0, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}
