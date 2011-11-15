; RUN: llc < %s -march=x86-64 | FileCheck %s
; Verify that we are using the efficient uitofp --> sitofp lowering illustrated
; by the compiler_rt implementation of __floatundisf.
; <rdar://problem/8493982>

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; FIXME: This test could generate this code:
;
; ## BB#0:                                ## %entry
; 	testq	%rdi, %rdi
; 	jns	LBB0_2
; ## BB#1:
; 	movq	%rdi, %rax
; 	shrq	%rax
; 	andq	$1, %rdi
; 	orq	%rax, %rdi
; 	cvtsi2ssq	%rdi, %xmm0
; 	addss	%xmm0, %xmm0
; 	ret
; LBB0_2:                                 ## %entry
; 	cvtsi2ssq	%rdi, %xmm0
; 	ret
;
; The blocks come from lowering:
;
;   %vreg7<def> = CMOV_FR32 %vreg6<kill>, %vreg5<kill>, 15, %EFLAGS<imp-use>; FR32:%vreg7,%vreg6,%vreg5
;
; If the instruction had an EFLAGS<kill> flag, it wouldn't need to mark EFLAGS
; as live-in on the new blocks, and machine sinking would be able to sink
; everything below the test.

; CHECK: shrq
; CHECK: andq
; CHECK-NEXT: orq
; CHECK: testq %rdi, %rdi
; CHECK-NEXT: jns LBB0_2
; CHECK: cvtsi2ss
; CHECK: LBB0_2
; CHECK-NEXT: cvtsi2ss
define float @test(i64 %a) {
entry:
  %b = uitofp i64 %a to float
  ret float %b
}
