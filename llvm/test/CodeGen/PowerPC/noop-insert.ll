; RUN: llc < %s -march=ppc32 -mcpu=g4 -noop-insertion | FileCheck %s
; RUN: llc < %s -march=ppc32 -mcpu=g4 -noop-insertion -rng-seed=1 | FileCheck %s --check-prefix=SEED1
; RUN: llc < %s -march=ppc32 -mcpu=g4 -noop-insertion -noop-insertion-percentage=100 | FileCheck %s --check-prefix=100PERCENT

; This test case checks that NOOPs are inserted correctly for PowerPC.

; It just happens that with a default percentage of 25% and seed=0,
; no NOOPs are inserted.
; CHECK: mullw
; CHECK-NEXT: add
; CHECK-NEXT: blr

; SEED1: nop
; SEED1-NEXT: mullw
; SEED1-NEXT: add
; SEED1-NEXT: nop
; SEED1-NEXT: blr

; 100PERCENT: nop
; 100PERCENT-NEXT: mullw
; 100PERCENT-NEXT: nop
; 100PERCENT-NEXT: add
; 100PERCENT-NEXT: nop
; 100PERCENT-NEXT: blr

define i32 @test1(i32 %x, i32 %y, i32 %z) {
entry:
    %tmp = mul i32 %x, %y
    %tmp2 = add i32 %tmp, %z
    ret i32 %tmp2
}
