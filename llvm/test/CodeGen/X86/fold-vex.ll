; Use CPU parameters to ensure that a CPU-specific attribute is not overriding the AVX definition.

; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=corei7-avx | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=btver2 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx | FileCheck %s

; No need to load unaligned operand from memory using an explicit instruction with AVX.
; The operand should be folded into the AND instr.

define <4 x i32> @test1(<4 x i32>* %p0, <4 x i32> %in1) nounwind {
  %in0 = load <4 x i32>* %p0, align 2
  %a = and <4 x i32> %in0, %in1
  ret <4 x i32> %a

; CHECK-LABEL: @test1
; CHECK-NOT:   vmovups
; CHECK:       vandps (%rdi), %xmm0, %xmm0
; CHECK-NEXT:  ret
}

