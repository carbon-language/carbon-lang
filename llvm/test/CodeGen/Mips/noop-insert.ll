; RUN: llc < %s -march=mips -noop-insertion | FileCheck %s
; RUN: llc < %s -march=mips -noop-insertion -rng-seed=1 | FileCheck %s --check-prefix=SEED1
; RUN: llc < %s -march=mips -noop-insertion -noop-insertion-percentage=100 | FileCheck %s --check-prefix=100PERCENT

; This test case checks that NOOPs are inserted correctly for MIPS.

; It just happens that with a default percentage of 25% and seed=0,
; no NOOPs are inserted.
; CHECK: mul
; CHECK-NEXT: jr

; SEED1: nop
; SEED1-NEXT: mul
; SEED1-NEXT: jr

; 100PERCENT: nop
; 100PERCENT-NEXT: mul
; 100PERCENT-NEXT: nop
; 100PERCENT-NEXT: jr

define i32 @test1(i32 %x, i32 %y, i32 %z) {
entry:
    %tmp = mul i32 %x, %y
    %tmp2 = add i32 %tmp, %z
    ret i32 %tmp2
}
