; RUN: opt < %s -analyze -block-freq | FileCheck %s

; A loop with multiple exits isn't irreducible.  It should be handled
; correctly.
;
; CHECK-LABEL: Printing analysis {{.*}} for function 'multiexit':
; CHECK-NEXT: block-frequency-info: multiexit
define void @multiexit(i1 %x) {
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
entry:
  br label %loop.1

; CHECK-NEXT: loop.1: float = 2.0,
loop.1:
  br i1 %x, label %exit.1, label %loop.2, !prof !0

; CHECK-NEXT: loop.2: float = 1.75,
loop.2:
  br i1 %x, label %exit.2, label %loop.1, !prof !1

; CHECK-NEXT: exit.1: float = 0.25,
exit.1:
  br label %return

; CHECK-NEXT: exit.2: float = 0.75,
exit.2:
  br label %return

; CHECK-NEXT: return: float = 1.0, int = [[ENTRY]]
return:
  ret void
}

!0 = metadata !{metadata !"branch_weights", i32 1, i32 7}
!1 = metadata !{metadata !"branch_weights", i32 3, i32 4}

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
define void @multientry(i1 %x) {
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
entry:
  br i1 %x, label %c1, label %c2, !prof !2

; This is like a single-line XFAIL (see above).
; CHECK-NEXT: c1:
; CHECK-NOT: float = 2.142857{{[0-9]*}},
c1:
  br i1 %x, label %c2, label %exit, !prof !2

; This is like a single-line XFAIL (see above).
; CHECK-NEXT: c2:
; CHECK-NOT: float = 1.857142{{[0-9]*}},
c2:
  br i1 %x, label %c1, label %exit, !prof !2

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
define void @crossloops(i2 %x) {
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
entry:
  switch i2 %x, label %exit [ i2 1, label %c1
                              i2 2, label %c2 ], !prof !3

; This is like a single-line XFAIL (see above).
; CHECK-NEXT: c1:
; CHECK-NOT: float = 1.0,
c1:
  switch i2 %x, label %exit [ i2 1, label %c1
                              i2 2, label %c2 ], !prof !3

; This is like a single-line XFAIL (see above).
; CHECK-NEXT: c2:
; CHECK-NOT: float = 1.0,
c2:
  switch i2 %x, label %exit [ i2 1, label %c1
                              i2 2, label %c2 ], !prof !3

; We still shouldn't lose any frequency.
; CHECK-NEXT: exit: float = 1.0, int = [[ENTRY]]
exit:
  ret void
}

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
