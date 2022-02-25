; RUN: opt < %s -analyze -enable-new-pm=0 -scalar-evolution -scalar-evolution-max-iterations=0  -scalar-evolution-classify-expressions=0  | FileCheck %s
; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" -scalar-evolution-max-iterations=0  -scalar-evolution-classify-expressions=0  2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @max_backedge_taken_count_by_wrapping1_nsw_nuw(i8 %N, i8* %ptr) {
; CHECK-LABEL: Determining loop execution counts for: @max_backedge_taken_count_by_wrapping1_nsw_nuw
; CHECK-NEXT:  Loop %loop: backedge-taken count is (%N /u 4)
; CHECK-NEXT:  Loop %loop: max backedge-taken count is 63
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is (%N /u 4)
;
entry:
  br label %loop

loop:
  %iv = phi i8 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr i8, i8* %ptr, i8 %iv
  store i8 %iv, i8* %gep
  %iv.next = add nuw nsw i8 %iv, 4
  %ec = icmp ne i8 %iv, %N
  br i1 %ec, label %loop, label %exit

exit:
  ret void
}

define void @max_backedge_taken_count_by_wrapping1_nuw(i8 %N, i8* %ptr) {
; CHECK-LABEL: Determining loop execution counts for: @max_backedge_taken_count_by_wrapping1_nuw
; CHECK-NEXT:  Loop %loop: backedge-taken count is (%N /u 4)
; CHECK-NEXT:  Loop %loop: max backedge-taken count is 63
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is (%N /u 4)
;
entry:
  br label %loop

loop:
  %iv = phi i8 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr i8, i8* %ptr, i8 %iv
  store i8 %iv, i8* %gep
  %iv.next = add nuw i8 %iv, 4
  %ec = icmp ne i8 %iv, %N
  br i1 %ec, label %loop, label %exit

exit:
  ret void
}

define void @max_backedge_taken_count_by_wrapping2_nsw_nuw(i8 %N, i8* %ptr) {
; CHECK-LABEL: Determining loop execution counts for: @max_backedge_taken_count_by_wrapping2
; CHECK-NEXT:  Loop %loop: backedge-taken count is ((-64 + %N) /u 4)
; CHECK-NEXT:  Loop %loop: max backedge-taken count is 63
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is ((-64 + %N) /u 4)
;
entry:
  br label %loop

loop:
  %iv = phi i8 [ 64, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr i8, i8* %ptr, i8 %iv
  store i8 %iv, i8* %gep
  %iv.next = add nuw nsw i8 %iv, 4
  %ec = icmp ne i8 %iv, %N
  br i1 %ec, label %loop, label %exit

exit:
  ret void
}

define void @max_backedge_taken_count_by_wrapping2_nuw(i8 %N, i8* %ptr) {
; CHECK-LABEL: Determining loop execution counts for: @max_backedge_taken_count_by_wrapping2
; CHECK-NEXT:  Loop %loop: backedge-taken count is ((-64  + %N) /u 4)
; CHECK-NEXT:  Loop %loop: max backedge-taken count is 63
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is ((-64 + %N) /u 4)
;
entry:
  br label %loop

loop:
  %iv = phi i8 [ 64, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr i8, i8* %ptr, i8 %iv
  store i8 %iv, i8* %gep
  %iv.next = add nuw i8 %iv, 4
  %ec = icmp ne i8 %iv, %N
  br i1 %ec, label %loop, label %exit

exit:
  ret void
}
