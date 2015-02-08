; RUN: opt -analyze -scalar-evolution < %s | FileCheck %s

; An example run where SCEV(%postinc)->getStart() may overflow:
;
; %start = INT_SMAX
; %low.limit = INT_SMIN
; %high.limit = < not used >
;
; >> entry:
;  %postinc.start = INT_SMIN
;
; >> loop:
;  %idx = %start 
;  %postinc = INT_SMIN
;  %postinc.inc = INT_SMIN + 1
;  %postinc.sext = sext(INT_SMIN) = i64 INT32_SMIN
;  %break.early = INT_SMIN `slt` INT_SMIN = false
;  br i1 false, ___,  %early.exit
;
; >> early.exit:
;  ret i64 INT32_SMIN


define i64 @bad.0(i32 %start, i32 %low.limit, i32 %high.limit) {
; CHECK-LABEL: Classifying expressions for: @bad.0
 entry:
  %postinc.start = add i32 %start, 1
  br label %loop

 loop:
  %idx = phi i32 [ %start, %entry ], [ %idx.inc, %continue ]
  %postinc = phi i32 [ %postinc.start, %entry ], [ %postinc.inc, %continue ]
  %postinc.inc = add nsw i32 %postinc, 1
  %postinc.sext = sext i32 %postinc to i64
; CHECK:  %postinc.sext = sext i32 %postinc to i64
; CHECK-NEXT:  -->  {(sext i32 (1 + %start) to i64),+,1}<nsw><%loop>
  %break.early = icmp slt i32 %postinc, %low.limit
  br i1 %break.early, label %continue, label %early.exit

 continue:
  %idx.inc = add nsw i32 %idx, 1
  %cmp = icmp slt i32 %idx.inc, %high.limit
  br i1 %cmp, label %loop, label %exit

 exit:
  ret i64 0

 early.exit:
  ret i64 %postinc.sext
}

define i64 @bad.1(i32 %start, i32 %low.limit, i32 %high.limit, i1* %unknown) {
; CHECK-LABEL: Classifying expressions for: @bad.1
 entry:
  %postinc.start = add i32 %start, 1
  br label %loop

 loop:
  %idx = phi i32 [ %start, %entry ], [ %idx.inc, %continue ], [ %idx.inc, %continue.1 ]
  %postinc = phi i32 [ %postinc.start, %entry ], [ %postinc.inc, %continue ], [ %postinc.inc, %continue.1 ]
  %postinc.inc = add nsw i32 %postinc, 1
  %postinc.sext = sext i32 %postinc to i64
; CHECK:  %postinc.sext = sext i32 %postinc to i64
; CHECK-NEXT:  -->  {(sext i32 (1 + %start) to i64),+,1}<nsw><%loop>
  %break.early = icmp slt i32 %postinc, %low.limit
  br i1 %break.early, label %continue.1, label %early.exit

 continue.1:
  %cond = load volatile i1* %unknown
  %idx.inc = add nsw i32 %idx, 1
  br i1 %cond, label %loop, label %continue

 continue:
  %cmp = icmp slt i32 %idx.inc, %high.limit
  br i1 %cmp, label %loop, label %exit

 exit:
  ret i64 0

 early.exit:
  ret i64 %postinc.sext
}


; WARNING: FIXME: it is safe to make the inference demonstrated here
; only if we assume `add nsw` has undefined behavior if the result
; sign-overflows; and this interpretation is stronger than what most
; of LLVM assumes.  This test here only serves as a documentation of
; current behavior and will need to be revisited once we've decided
; upon a consistent semantics for nsw (and nuw) arithetic operations.
;
define i64 @good(i32 %start, i32 %low.limit, i32 %high.limit) {
; CHECK-LABEL: Classifying expressions for: @good
 entry:
  %postinc.start = add i32 %start, 1
  br label %loop

 loop:
  %idx = phi i32 [ %start, %entry ], [ %idx.inc, %loop ]
  %postinc = phi i32 [ %postinc.start, %entry ], [ %postinc.inc, %loop ]
  %postinc.inc = add nsw i32 %postinc, 1
  %postinc.sext = sext i32 %postinc to i64
; CHECK: %postinc.sext = sext i32 %postinc to i64
; CHECK-NEXT: -->  {(1 + (sext i32 %start to i64)),+,1}<nsw><%loop>

  %break.early = icmp slt i32 %postinc, %low.limit
  %idx.inc = add nsw i32 %idx, 1
  %cmp = icmp slt i32 %idx.inc, %high.limit
  br i1 %cmp, label %loop, label %exit

 exit:
  ret i64 0

 early.exit:
  ret i64 %postinc.sext
}
