; REQUIRES: disabled

; RUN: llc < %s -mtriple=x86_64-linux -noop-insertion | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-linux -noop-insertion -rng-seed=1 | FileCheck %s --check-prefix=SEED1
; RUN: llc < %s -mtriple=x86_64-linux -noop-insertion -rng-seed=20 | FileCheck %s --check-prefix=SEED2
; RUN: llc < %s -mtriple=x86_64-linux -noop-insertion -rng-seed=500 | FileCheck %s --check-prefix=SEED3

; RUN: llc < %s -march=x86 -noop-insertion | FileCheck %s --check-prefix=x86_32

; This test case checks that NOOPs are inserted, and that the RNG seed
; affects both the placement (position of imull) and choice of these NOOPs.

; It just happens that with a default percentage of 25% and seed=0,
; no NOOPs are inserted.
; CHECK: imull
; CHECK-NEXT: leal
; CHECK-NEXT: retq
; CHECK-NOT: nop

; SEED1: leaq (%rsi), %rsi
; SEED1-NEXT: imull
; SEED1-NEXT: leal
; SEED1-NEXT: retq

; SEED2: imull
; SEED2-NEXT: movq %rsp, %rsp
; SEED2-NEXT: leal
; SEED2-NEXT: retq

; SEED3: imull
; SEED3-NEXT: movq %rsp, %rsp
; SEED3-NEXT: leal
; SEED3-NEXT: leaq (%rdi), %rdi
; SEED3-NEXT: retq

; The operand of the following is used to distinguish from a movl NOOP
; x86_32: movl 4(%esp),
; x86_32-NEXT: imull
; x86_32-NEXT: addl
; x86_32-NEXT: movl %esp, %esp
; x86_32-NEXT: retl

define i32 @test1(i32 %x, i32 %y, i32 %z) {
entry:
    %tmp = mul i32 %x, %y
    %tmp2 = add i32 %tmp, %z
    ret i32 %tmp2
}
