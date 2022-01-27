; RUN: llc -march=avr -print-after=finalize-isel -cgp-freq-ratio-to-skip-merge=10 < %s 2>&1 | FileCheck %s

; Because `switch` seems to trigger Machine Basic Blocks to be ordered
; in a different order than they were constructed, this exposes an
; error in the `expand-isel-pseudos` pass. Specifically, it thought we
; could always fallthrough to a newly-constructed MBB. However,
; there's no guarantee that either of the constructed MBBs need to
; occur immediately after the currently-focused one!
;
; This issue manifests in a CFG that looks something like this:
;
; %bb.2.finish:
;     successors: %bb.5(?%) %bb.6(?%)
;     Predecessors according to CFG: %bb.0 %bb.1
;         %0 = PHI %3, <%bb.0>, %5, <%bb.1>
;         %7 = LDIRdK 2
;         %8 = LDIRdK 1
;         CPRdRr %2, %0, implicit-def %SREG
;         BREQk <%bb.6>, implicit %SREG
;
; The code assumes it the fallthrough block after this is %bb.5, but
; it's actually %bb.3! To be proper, there should be an unconditional
; jump tying this block to %bb.5.

define i8 @select_must_add_unconditional_jump(i8 %arg0, i8 %arg1) unnamed_addr {
entry-block:
  switch i8 %arg0, label %dead [
    i8 0, label %zero
    i8 1, label %one
  ]

zero:
  br label %finish

one:
  br label %finish

finish:
  %predicate = phi i8 [ 50, %zero ], [ 100, %one ]
  %is_eq = icmp eq i8 %arg1, %predicate
  %result = select i1 %is_eq, i8 1, i8 2
  ret i8 %result

dead:
  ret i8 0
}

; This check may be a bit brittle, but the important thing is that the
; basic block containing `select` needs to contain explicit jumps to
; both successors.

; CHECK: bb.2.finish:
; CHECK: successors:
; CHECK: BREQk [[BRANCHED:%bb.[0-9]+]]
; CHECK: RJMPk [[DIRECT:%bb.[0-9]+]]
; CHECK-SAME-DAG: {{.*}}[[BRANCHED]]
; CHECK-SAME-DAG: {{.*}}[[DIRECT]]
; CHECK: bb.3.dead:
