; This test ensures that IVUsers works correctly in the legacy pass manager
; without LCSSA and in the specific ways that some of its users (LSR) require.
;
; FIXME: We need some way to match the precision here in the new PM where loop
; passes *always* work on LCSSA. This should stop using a different set of
; checks at that point.

; RUN: opt < %s -analyze -iv-users -enable-new-pm=0 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-LCSSA
; RUN: opt < %s -disable-output -passes='print<iv-users>' 2>&1 | FileCheck %s

; Provide legal integer types.
target datalayout = "n8:16:32:64"

; The value of %r is dependent on a polynomial iteration expression.
;
; CHECK-LABEL: IV Users for loop %foo.loop
; CHECK-NO-LCSSA: {1,+,3,+,2}<%foo.loop>
define i64 @foo(i64 %n) {
entry:
  br label %foo.loop

foo.loop:
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %foo.loop ]
  %indvar.next = add i64 %indvar, 1
  %c = icmp eq i64 %indvar.next, %n
  br i1 %c, label %exit, label %foo.loop

exit:
  %r = mul i64 %indvar.next, %indvar.next
  ret i64 %r
}

; PR15470: LSR miscompile. The test1 function should return '1'.
; It is valid to fold SCEVUnknown into the recurrence because it
; was defined before the loop.
;
; SCEV does not know how to denormalize chained recurrences, so make
; sure they aren't marked as post-inc users.
;
; CHECK-LABEL: IV Users for loop %test1.loop
; CHECK-NO-LCSSA: %sext.us = {0,+,(16777216 + (-16777216 * %sub.us)<nuw><nsw>)<nuw><nsw>,+,33554432}<%test1.loop> (post-inc with loop %test1.loop) in    %f = ashr i32 %sext.us, 24
define i32 @test1(i1 %cond) {
entry:
  %sub.us = select i1 %cond, i32 0, i32 0
  br label %test1.loop

test1.loop:
  %inc1115.us = phi i32 [ 0, %entry ], [ %inc11.us, %test1.loop ]
  %inc11.us = add nsw i32 %inc1115.us, 1
  %cmp.us = icmp slt i32 %inc11.us, 2
  br i1 %cmp.us, label %test1.loop, label %for.end

for.end:
  %tobool.us = icmp eq i32 %inc1115.us, 0
  %mul.us = shl i32 %inc1115.us, 24
  %sub.cond.us = sub nsw i32 %inc1115.us, %sub.us
  %sext.us = mul i32 %mul.us, %sub.cond.us
  %f = ashr i32 %sext.us, 24
  br label %exit

exit:
  ret i32 %f
}

; PR15470: LSR miscompile. The test2 function should return '1'.
; It is illegal to fold SCEVUnknown (sext.us) into the recurrence
; because it is defined after the loop where this recurrence belongs.
;
; SCEV does not know how to denormalize chained recurrences, so make
; sure they aren't marked as post-inc users.
;
; CHECK-LABEL: IV Users for loop %test2.loop
; CHECK-NO-LCSSA: %sub.cond.us = ((-1 * %sub.us)<nuw><nsw> + {0,+,1}<nuw><nsw><%test2.loop>) (post-inc with loop %test2.loop) in    %sext.us = mul i32 %mul.us, %sub.cond.us
define i32 @test2() {
entry:
  br label %test2.loop

test2.loop:
  %inc1115.us = phi i32 [ 0, %entry ], [ %inc11.us, %test2.loop ]
  %inc11.us = add nsw i32 %inc1115.us, 1
  %cmp.us = icmp slt i32 %inc11.us, 2
  br i1 %cmp.us, label %test2.loop, label %for.end

for.end:
  %tobool.us = icmp eq i32 %inc1115.us, 0
  %sub.us = select i1 %tobool.us, i32 0, i32 0
  %mul.us = shl i32 %inc1115.us, 24
  %sub.cond.us = sub nsw i32 %inc1115.us, %sub.us
  %sext.us = mul i32 %mul.us, %sub.cond.us
  %f = ashr i32 %sext.us, 24
  br label %exit

exit:
  ret i32 %f
}
