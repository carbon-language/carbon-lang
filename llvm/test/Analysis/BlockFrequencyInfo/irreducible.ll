; RUN: opt < %s -analyze -block-freq | FileCheck %s

; A loop with multiple exits should be handled correctly.
;
; CHECK-LABEL: Printing analysis {{.*}} for function 'multiexit':
; CHECK-NEXT: block-frequency-info: multiexit
define void @multiexit(i32 %a) {
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
entry:
  br label %loop.1

; CHECK-NEXT: loop.1: float = 1.333{{3*}},
loop.1:
  %i = phi i32 [ 0, %entry ], [ %inc.2, %loop.2 ]
  call void @f(i32 %i)
  %inc.1 = add i32 %i, 1
  %cmp.1 = icmp ugt i32 %inc.1, %a
  br i1 %cmp.1, label %exit.1, label %loop.2, !prof !0

; CHECK-NEXT: loop.2: float = 0.666{{6*7}},
loop.2:
  call void @g(i32 %inc.1)
  %inc.2 = add i32 %inc.1, 1
  %cmp.2 = icmp ugt i32 %inc.2, %a
  br i1 %cmp.2, label %exit.2, label %loop.1, !prof !1

; CHECK-NEXT: exit.1: float = 0.666{{6*7}},
exit.1:
  call void @h(i32 %inc.1)
  br label %return

; CHECK-NEXT: exit.2: float = 0.333{{3*}},
exit.2:
  call void @i(i32 %inc.2)
  br label %return

; CHECK-NEXT: return: float = 1.0, int = [[ENTRY]]
return:
  ret void
}

declare void @f(i32 %x)
declare void @g(i32 %x)
declare void @h(i32 %x)
declare void @i(i32 %x)

!0 = metadata !{metadata !"branch_weights", i32 3, i32 3}
!1 = metadata !{metadata !"branch_weights", i32 5, i32 5}

; The current BlockFrequencyInfo algorithm doesn't handle multiple entrances
; into a loop very well.  The frequencies assigned to blocks in the loop are
; predictable (and not absurd), but also not correct and therefore not worth
; testing.
;
; There are two testcases below.
;
; For each testcase, I use a CHECK-NEXT/NOT combo like an XFAIL with the
; granularity of a single check.  If/when this behaviour is fixed, we'll know
; about it, and the test should be updated.
;
; Testcase #1
; ===========
;
; In this case c1 and c2 should have frequencies of 15/7 and 13/7,
; respectively.  To calculate this, consider assigning 1.0 to entry, and
; distributing frequency iteratively (to infinity).  At the first iteration,
; entry gives 3/4 to c1 and 1/4 to c2.  At every step after, c1 and c2 give 3/4
; of what they have to each other.  Somehow, all of it comes out to exit.
;
;       c1 = 3/4 + 1/4*3/4 + 3/4*3^2/4^2 + 1/4*3^3/4^3 + 3/4*3^3/4^3 + ...
;       c2 = 1/4 + 3/4*3/4 + 1/4*3^2/4^2 + 3/4*3^3/4^3 + 1/4*3^3/4^3 + ...
;
; Simplify by splitting up the odd and even terms of the series and taking out
; factors so that the infite series matches:
;
;       c1 =  3/4 *(9^0/16^0 + 9^1/16^1 + 9^2/16^2 + ...)
;          +  3/16*(9^0/16^0 + 9^1/16^1 + 9^2/16^2 + ...)
;       c2 =  1/4 *(9^0/16^0 + 9^1/16^1 + 9^2/16^2 + ...)
;          +  9/16*(9^0/16^0 + 9^1/16^1 + 9^2/16^2 + ...)
;
;       c1 = 15/16*(9^0/16^0 + 9^1/16^1 + 9^2/16^2 + ...)
;       c2 = 13/16*(9^0/16^0 + 9^1/16^1 + 9^2/16^2 + ...)
;
; Since this geometric series sums to 16/7:
;
;       c1 = 15/7
;       c2 = 13/7
;
; If we treat c1 and c2 as members of the same loop, the exit frequency of the
; loop as a whole is 1/4, so the loop scale should be 4.  Summing c1 and c2
; gives 28/7, or 4.0, which is nice confirmation of the math above.
;
; However, assuming c1 precedes c2 in reverse post-order, the current algorithm
; returns 3/4 and 13/16, respectively.  LoopInfo ignores edges between loops
; (and doesn't see any loops here at all), and -block-freq ignores the
; irreducible edge from c2 to c1.
;
; CHECK-LABEL: Printing analysis {{.*}} for function 'multientry':
; CHECK-NEXT: block-frequency-info: multientry
define void @multientry(i32 %a) {
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
entry:
  %choose = call i32 @choose(i32 %a)
  %compare = icmp ugt i32 %choose, %a
  br i1 %compare, label %c1, label %c2, !prof !2

; This is like a single-line XFAIL (see above).
; CHECK-NEXT: c1:
; CHECK-NOT: float = 2.142857{{[0-9]*}},
c1:
  %i1 = phi i32 [ %a, %entry ], [ %i2.inc, %c2 ]
  %i1.inc = add i32 %i1, 1
  %choose1 = call i32 @choose(i32 %i1)
  %compare1 = icmp ugt i32 %choose1, %a
  br i1 %compare1, label %c2, label %exit, !prof !2

; This is like a single-line XFAIL (see above).
; CHECK-NEXT: c2:
; CHECK-NOT: float = 1.857142{{[0-9]*}},
c2:
  %i2 = phi i32 [ %a, %entry ], [ %i1.inc, %c1 ]
  %i2.inc = add i32 %i2, 1
  %choose2 = call i32 @choose(i32 %i2)
  %compare2 = icmp ugt i32 %choose2, %a
  br i1 %compare2, label %c1, label %exit, !prof !2

; We still shouldn't lose any frequency.
; CHECK-NEXT: exit: float = 1.0, int = [[ENTRY]]
exit:
  ret void
}

; Testcase #2
; ===========
;
; In this case c1 and c2 should be treated as equals in a single loop.  The
; exit frequency is 1/3, so the scaling factor for the loop should be 3.0.  The
; loop is entered 2/3 of the time, and c1 and c2 split the total loop frequency
; evenly (1/2), so they should each have frequencies of 1.0 (3.0*2/3*1/2).
; Another way of computing this result is by assigning 1.0 to entry and showing
; that c1 and c2 should accumulate frequencies of:
;
;       1/3   +   2/9   +   4/27  +   8/81  + ...
;     2^0/3^1 + 2^1/3^2 + 2^2/3^3 + 2^3/3^4 + ...
;
; At the first step, c1 and c2 each get 1/3 of the entry.  At each subsequent
; step, c1 and c2 each get 1/3 of what's left in c1 and c2 combined.  This
; infinite series sums to 1.
;
; However, assuming c1 precedes c2 in reverse post-order, the current algorithm
; returns 1/2 and 3/4, respectively.  LoopInfo ignores edges between loops (and
; treats c1 and c2 as self-loops only), and -block-freq ignores the irreducible
; edge from c2 to c1.
;
; Below I use a CHECK-NEXT/NOT combo like an XFAIL with the granularity of a
; single check.  If/when this behaviour is fixed, we'll know about it, and the
; test should be updated.
;
; CHECK-LABEL: Printing analysis {{.*}} for function 'crossloops':
; CHECK-NEXT: block-frequency-info: crossloops
define void @crossloops(i32 %a) {
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
entry:
  %choose = call i32 @choose(i32 %a)
  switch i32 %choose, label %exit [ i32 1, label %c1
                                    i32 2, label %c2 ], !prof !3

; This is like a single-line XFAIL (see above).
; CHECK-NEXT: c1:
; CHECK-NOT: float = 1.0,
c1:
  %i1 = phi i32 [ %a, %entry ], [ %i1.inc, %c1 ], [ %i2.inc, %c2 ]
  %i1.inc = add i32 %i1, 1
  %choose1 = call i32 @choose(i32 %i1)
  switch i32 %choose1, label %exit [ i32 1, label %c1
                                     i32 2, label %c2 ], !prof !3

; This is like a single-line XFAIL (see above).
; CHECK-NEXT: c2:
; CHECK-NOT: float = 1.0,
c2:
  %i2 = phi i32 [ %a, %entry ], [ %i1.inc, %c1 ], [ %i2.inc, %c2 ]
  %i2.inc = add i32 %i2, 1
  %choose2 = call i32 @choose(i32 %i2)
  switch i32 %choose2, label %exit [ i32 1, label %c1
                                     i32 2, label %c2 ], !prof !3

; We still shouldn't lose any frequency.
; CHECK-NEXT: exit: float = 1.0, int = [[ENTRY]]
exit:
  ret void
}

declare i32 @choose(i32)

!2 = metadata !{metadata !"branch_weights", i32 3, i32 1}
!3 = metadata !{metadata !"branch_weights", i32 2, i32 2, i32 2}

; A reducible loop with irreducible control flow inside should still have
; correct exit frequency.
;
; CHECK-LABEL: Printing analysis {{.*}} for function 'loop_around_irreducible':
; CHECK-NEXT: block-frequency-info: loop_around_irreducible
define void @loop_around_irreducible(i1 %x) {
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
entry:
  br label %loop

; CHECK-NEXT: loop: float = [[HEAD:[0-9.]+]], int = [[HEADINT:[0-9]+]]
loop:
  br i1 %x, label %left, label %right

; CHECK-NEXT: left:
left:
  br i1 %x, label %right, label %loop.end

; CHECK-NEXT: right:
right:
  br i1 %x, label %left, label %loop.end

; CHECK-NEXT: loop.end: float = [[HEAD]], int = [[HEADINT]]
loop.end:
  br i1 %x, label %loop, label %exit

; CHECK-NEXT: float = 1.0, int = [[ENTRY]]
exit:
  ret void
}
