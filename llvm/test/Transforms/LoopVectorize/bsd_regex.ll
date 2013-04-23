; RUN: opt -S -loop-vectorize -dce -instcombine -force-vector-width=2 -force-vector-unroll=2 < %s | FileCheck %s

;PR 15830.

;CHECK: foo
; When scalarizing stores we need to preserve the original order.
; Make sure that we are extracting in the correct order (0101, and not 0011).
;CHECK: extractelement <2 x i64> {{.*}}, i32 0
;CHECK: extractelement <2 x i64> {{.*}}, i32 1
;CHECK: extractelement <2 x i64> {{.*}}, i32 0
;CHECK: extractelement <2 x i64> {{.*}}, i32 1
;CHECK: store
;CHECK: store
;CHECK: store
;CHECK: store
;CHECK: ret

define i32 @foo(i32* nocapture %A) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = shl nsw i64 %indvars.iv, 2
  %arrayidx = getelementptr inbounds i32* %A, i64 %0
  store i32 4, i32* %arrayidx, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 10000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 undef
}


