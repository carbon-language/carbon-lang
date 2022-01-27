; RUN: opt -loop-unroll -S -mtriple aarch64 -mcpu=cortex-a57 %s | FileCheck %s -check-prefix=UNROLL
; RUN: opt -loop-unroll -unroll-max-upperbound=0 -S -mtriple aarch64 -mcpu=cortex-a57 %s | FileCheck %s -check-prefix=NOUNROLL

; This IR comes from this C code:
;
;   for (int i = 0; i < 4; i++) {
;     if (src[i] == 1) {
;       *dst = i;
;       break;
;     }
;   }
;
; This test is meant to check that this loop is unrolled into four iterations.
; Note that the load on the last iteration is dead and thus doesn't appear in
; the output.

; UNROLL-LABEL: @test
; UNROLL: load i32, i32*
; UNROLL: load i32, i32*
; UNROLL: load i32, i32*
; UNROLL-NOT: load i32, i32*
; NOUNROLL-LABEL: @test
; NOUNROLL: load i32, i32*
; NOUNROLL-NOT: load i32, i32*

define void @test(i32* %dst, i32* %src) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %0 = sext i32 %i to i64
  %1 = getelementptr inbounds i32, i32* %src, i64 %0
  %2 = load i32, i32* %1
  %inc = add nsw i32 %i, 1
  %cmp1 = icmp slt i32 %inc, 4
  %cmp3 = icmp eq i32 %2, 1 
  %or.cond = and i1 %cmp3, %cmp1
  br i1 %or.cond, label %for.body, label %exit

exit:                                          ; preds = %for.body
  store i32 %i, i32* %dst
  ret void
}
