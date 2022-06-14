; REQUIRES: asserts
; RUN: opt < %s -S -loop-unroll -unroll-runtime=true -unroll-allow-remainder=true -unroll-count=4

; Make sure that the runtime unroll does not break with a non-exiting latch.
define i32 @test(i32* %a, i32* %b, i32* %c, i64 %n) {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc, %while.body ]
  %cmp = icmp slt i64 %i.0, %n
  br i1 %cmp, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %i.0
  %0 = load i32, i32* %arrayidx
  %arrayidx1 = getelementptr inbounds i32, i32* %c, i64 %i.0
  %1 = load i32, i32* %arrayidx1
  %mul = mul nsw i32 %0, %1
  %arrayidx2 = getelementptr inbounds i32, i32* %a, i64 %i.0
  store i32 %mul, i32* %arrayidx2
  %inc = add nsw i64 %i.0, 1
  br label %while.cond

while.end:                                        ; preds = %while.cond
  ret i32 0
}
