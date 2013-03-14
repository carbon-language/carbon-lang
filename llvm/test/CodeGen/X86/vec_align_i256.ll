; RUN: llc < %s -mcpu=corei7-avx | FileCheck %s 

; Make sure that we are not generating a movaps because the vector is aligned to 1.
;CHECK: @foo
;CHECK: xor
;CHECK-NEXT: vmovups
;CHECK-NEXT: ret
define void @foo() {
  store <16 x i16> zeroinitializer, <16 x i16>* undef, align 1
  ret void
}
