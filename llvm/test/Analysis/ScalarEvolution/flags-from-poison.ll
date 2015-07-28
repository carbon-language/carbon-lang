; RUN: opt < %s -S -analyze -scalar-evolution | FileCheck %s

; Positive and negative tests for inferring flags like nsw from
; reasoning about how a poison value from overflow would trigger
; undefined behavior.

define void @foo() {
  ret void
}

; Example where an add should get the nsw flag, so that a sext can be
; distributed over the add.
define void @test-add-nsw(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-nsw
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nsw>
  %index32 = add nsw i32 %i, %offset

; CHECK: %index64 =
; CHECK: --> {(sext i32 %offset to i64),+,1}<nsw>
  %index64 = sext i32 %index32 to i64

  %ptr = getelementptr inbounds float, float* %input, i64 %index64
  %nexti = add nsw i32 %i, 1
  %f = load float, float* %ptr, align 4
  call void @foo()
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Example where an add should get the nuw flag.
define void @test-add-nuw(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-nuw
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nuw>
  %index32 = add nuw i32 %i, %offset

  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %nexti = add nuw i32 %i, 1
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

; With no load to trigger UB from poison, we cannot infer nsw.
define void @test-add-no-load(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-no-load
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nw>
  %index32 = add nsw i32 %i, %offset

  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %nexti = add nuw i32 %i, 1
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

; The current code is only supposed to look at the loop header, so
; it should not infer nsw in this case, as that would require looking
; outside the loop header.
define void @test-add-not-header(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-not-header
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop2 ], [ 0, %entry ]
  br label %loop2
loop2:

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nw>
  %index32 = add nsw i32 %i, %offset

  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %nexti = add nsw i32 %i, 1
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Same thing as test-add-not-header, but in this case only the load
; instruction is outside the loop header.
define void @test-add-not-header2(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-not-header2
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop2 ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nw>
  %index32 = add nsw i32 %i, %offset

  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %nexti = add nsw i32 %i, 1
  br label %loop2
loop2:
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; The call instruction makes it not guaranteed that the add will be
; executed, since it could run forever or throw an exception, so we
; cannot assume that the UB is realized.
define void @test-add-call(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-call
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nw>
  call void @foo()
  %index32 = add nsw i32 %i, %offset

  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %nexti = add nsw i32 %i, 1
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Same issue as test-add-call, but this time the call is between the
; producer of poison and the load that consumes it.
define void @test-add-call2(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-call2
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nw>
  %index32 = add nsw i32 %i, %offset

  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %nexti = add nsw i32 %i, 1
  call void @foo()
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Without inbounds, GEP does not propagate poison in the very
; conservative approach used here.
define void @test-add-no-inbounds(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-no-inbounds
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nw>
  %index32 = add nsw i32 %i, %offset

  %ptr = getelementptr float, float* %input, i32 %index32
  %nexti = add nsw i32 %i, 1
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Multiplication by a non-zero constant propagates poison if there is
; a nuw or nsw flag on the multiplication.
define void @test-add-mul-propagates(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-mul-propagates
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nsw>
  %index32 = add nsw i32 %i, %offset

  %indexmul = mul nuw i32 %index32, 2
  %ptr = getelementptr inbounds float, float* %input, i32 %indexmul
  %nexti = add nsw i32 %i, 1
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Multiplication by a non-constant should not propagate poison in the
; very conservative approach used here.
define void @test-add-mul-no-propagation(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-mul-no-propagation
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nw>
  %index32 = add nsw i32 %i, %offset

  %indexmul = mul nsw i32 %index32, %offset
  %ptr = getelementptr inbounds float, float* %input, i32 %indexmul
  %nexti = add nsw i32 %i, 1
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Multiplication by a non-zero constant does not propagate poison
; without a no-wrap flag.
define void @test-add-mul-no-propagation2(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-mul-no-propagation2
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nw>
  %index32 = add nsw i32 %i, %offset

  %indexmul = mul i32 %index32, 2
  %ptr = getelementptr inbounds float, float* %input, i32 %indexmul
  %nexti = add nsw i32 %i, 1
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Division by poison triggers UB.
define void @test-add-div(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-div
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %j =
; CHECK: --> {%offset,+,1}<nsw>
  %j = add nsw i32 %i, %offset

  %q = sdiv i32 %numIterations, %j
  %nexti = add nsw i32 %i, 1
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Remainder of poison by non-poison divisor does not trigger UB.
define void @test-add-div2(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-div2
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %j =
; CHECK: --> {%offset,+,1}<nw>
  %j = add nsw i32 %i, %offset

  %q = sdiv i32 %j, %numIterations
  %nexti = add nsw i32 %i, 1
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Store to poison address triggers UB.
define void @test-add-store(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-store
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nsw>
  %index32 = add nsw i32 %i, %offset

  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %nexti = add nsw i32 %i, 1
  store float 1.0, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Three sequential adds where the middle add should have nsw. There is
; a special case for sequential adds and this test covers that. We have to
; put the final add first in the program since otherwise the special case
; is not triggered, hence the strange basic block ordering.
define void @test-add-twice(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-twice
entry:
  br label %loop
loop2:
; CHECK: %seq =
; CHECK: --> {(2 + %offset),+,1}<nw>
  %seq = add nsw nuw i32 %index32, 1
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop

loop:
  %i = phi i32 [ %nexti, %loop2 ], [ 0, %entry ]

  %j = add nsw i32 %i, 1
; CHECK: %index32 =
; CHECK: --> {(1 + %offset),+,1}<nsw>
  %index32 = add nsw i32 %j, %offset

  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %nexti = add nsw i32 %i, 1
  store float 1.0, float* %ptr, align 4
  br label %loop2
exit:
  ret void
}
