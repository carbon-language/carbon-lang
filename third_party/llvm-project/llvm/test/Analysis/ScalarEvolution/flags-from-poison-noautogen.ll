; RUN: opt < %s -S -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s

; This file is conceptually part of flags-from-poison.ll except that the
; test does not successfully auto-update via utils/update_analysis_tests_checks.sh

; Subtraction of two recurrences. The addition in the SCEV that this
; maps to is NSW, but the negation of the RHS does not since that
; recurrence could be the most negative representable value.
define void @subrecurrences(i32 %outer_l, i32 %inner_l, i32 %val) {
; CHECK-LABEL: 'subrecurrences'
; CHECK-NEXT:  Classifying expressions for: @subrecurrences
; CHECK-NEXT:    %o_idx = phi i32 [ 0, %entry ], [ %o_idx.inc, %outer.be ]
; CHECK-NEXT:    --> {0,+,1}<nuw><nsw><%outer> U: [0,-2147483648) S: [0,-2147483648) Exits: %outer_l LoopDispositions: { %outer: Computable, %inner: Invariant }
; CHECK-NEXT:    %o_idx.inc = add nsw i32 %o_idx, 1
; CHECK-NEXT:    --> {1,+,1}<nuw><%outer> U: [1,0) S: [1,0) Exits: (1 + %outer_l) LoopDispositions: { %outer: Computable, %inner: Invariant }
; CHECK-NEXT:    %i_idx = phi i32 [ 0, %outer ], [ %i_idx.inc, %inner ]
; CHECK-NEXT:    --> {0,+,1}<nuw><nsw><%inner> U: [0,-2147483648) S: [0,-2147483648) Exits: %inner_l LoopDispositions: { %inner: Computable, %outer: Variant }
; CHECK-NEXT:    %i_idx.inc = add nsw i32 %i_idx, 1
; CHECK-NEXT:    --> {1,+,1}<nuw><%inner> U: [1,0) S: [1,0) Exits: (1 + %inner_l) LoopDispositions: { %inner: Computable, %outer: Variant }
; CHECK-NEXT:    %v = sub nsw i32 %i_idx, %o_idx.inc
; CHECK-NEXT:    -->
; NOTE: Line deleted from autogen output due to format confusing regex matcher
; CHECK-NEXT:    %forub = udiv i32 1, %v
; CHECK-NEXT:    -->
; NOTE: Line deleted from autogen output due to format confusing regex matcher
; CHECK-NEXT:  Determining loop execution counts for: @subrecurrences
; CHECK-NEXT:  Loop %inner: backedge-taken count is %inner_l
; CHECK-NEXT:  Loop %inner: max backedge-taken count is -1
; CHECK-NEXT:  Loop %inner: Predicated backedge-taken count is %inner_l
; CHECK-NEXT:   Predicates:
; CHECK:       Loop %inner: Trip multiple is 1
; CHECK-NEXT:  Loop %outer: backedge-taken count is %outer_l
; CHECK-NEXT:  Loop %outer: max backedge-taken count is -1
; CHECK-NEXT:  Loop %outer: Predicated backedge-taken count is %outer_l
; CHECK-NEXT:   Predicates:
; CHECK:       Loop %outer: Trip multiple is 1
;
entry:
  br label %outer

outer:
  %o_idx = phi i32 [ 0, %entry ], [ %o_idx.inc, %outer.be ]
  %o_idx.inc = add nsw i32 %o_idx, 1
  %cond = icmp eq i32 %o_idx, %val
  br i1 %cond, label %inner, label %outer.be

inner:
  %i_idx = phi i32 [ 0, %outer ], [ %i_idx.inc, %inner ]
  %i_idx.inc = add nsw i32 %i_idx, 1
  %v = sub nsw i32 %i_idx, %o_idx.inc
  %forub = udiv i32 1, %v
  %cond2 = icmp eq i32 %i_idx, %inner_l
  br i1 %cond2, label %outer.be, label %inner

outer.be:
  %cond3 = icmp eq i32 %o_idx, %outer_l
  br i1 %cond3, label %exit, label %outer

exit:
  ret void
}
