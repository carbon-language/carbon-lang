; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx | FileCheck %s


; rdar://13126763
; Expression "x + x*x" was mistakenly transformed into "x * 3.0f".

define float @test1(float %x) {
; CHECK-LABEL: test1:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    vmulss %xmm0, %xmm0, %xmm1
; CHECK-NEXT:    vaddss %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %t1 = fmul fast float %x, %x
  %t2 = fadd fast float %t1, %x
  ret float %t2
}

; (x + x) + x => x * 3.0
define float @test2(float %x) {
; CHECK-LABEL: test2:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    vmulss {{.*}}(%rip), %xmm0, %xmm0
; CHECK-NEXT:    retq
  %t1 = fadd fast float %x, %x
  %t2 = fadd fast float %t1, %x
  ret float %t2
}

; x + (x + x) => x * 3.0
define float @test3(float %x) {
; CHECK-LABEL: test3:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    vmulss {{.*}}(%rip), %xmm0, %xmm0
; CHECK-NEXT:    retq
  %t1 = fadd fast float %x, %x
  %t2 = fadd fast float %x, %t1
  ret float %t2
}

; (y + x) + x != x * 3.0
define float @test4(float %x, float %y) {
; CHECK-LABEL: test4:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    vaddss %xmm1, %xmm0, %xmm1
; CHECK-NEXT:    vaddss %xmm0, %xmm1, %xmm0
; CHECK-NEXT:    retq
  %t1 = fadd fast float %x, %y
  %t2 = fadd fast float %t1, %x
  ret float %t2
}

; rdar://13445387
; "x + x + x => 3.0 * x" should be disabled after legalization because
; Instruction-Selection doesn't know how to handle "3.0"
;
define float @test5(<4 x float> %x) {
; CHECK-LABEL: test5:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    vmulss {{.*}}(%rip), %xmm0, %xmm0
; CHECK-NEXT:    retq
  %splat = shufflevector <4 x float> %x, <4 x float> undef, <4 x i32> zeroinitializer
  %v1 = extractelement <4 x float> %splat, i32 1
  %v0 = extractelement <4 x float> %splat, i32 0
  %add1 = fadd reassoc nsz float %v0, %v1
  %v2 = extractelement <4 x float> %splat, i32 2
  %add2 = fadd reassoc nsz float %v2, %add1
  ret float %add2
}

