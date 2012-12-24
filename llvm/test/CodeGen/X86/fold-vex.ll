; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=corei7-avx | FileCheck %s

;CHECK: @test
; No need to load from memory. The operand will be loaded as part of th AND instr.
;CHECK-NOT: vmovaps
;CHECK: vandps
;CHECK: ret

define void @test1(<8 x i32>* %p0, <8 x i32> %in1) nounwind {
entry:
  %in0 = load <8 x i32>* %p0, align 2
  %a = and <8 x i32> %in0, %in1
  store <8 x i32> %a, <8 x i32>* undef
  ret void
}

