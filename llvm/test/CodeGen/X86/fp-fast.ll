; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=avx -enable-unsafe-fp-math < %s | FileCheck %s

define float @test1(float %a) {
; CHECK-LABEL: test1:
; CHECK:       # BB#0:
; CHECK-NEXT:    vmulss {{.*}}(%rip), %xmm0, %xmm0
; CHECK-NEXT:    retq
  %t1 = fadd float %a, %a
  %r = fadd float %t1, %t1
  ret float %r
}

define float @test2(float %a) {
; CHECK-LABEL: test2:
; CHECK:       # BB#0:
; CHECK-NEXT:    vmulss {{.*}}(%rip), %xmm0, %xmm0
; CHECK-NEXT:    retq
  %t1 = fmul float 4.0, %a
  %t2 = fadd float %a, %a
  %r = fadd float %t1, %t2
  ret float %r
}

define float @test3(float %a) {
; CHECK-LABEL: test3:
; CHECK:       # BB#0:
; CHECK-NEXT:    vmulss {{.*}}(%rip), %xmm0, %xmm0
; CHECK-NEXT:    retq
  %t1 = fmul float %a, 4.0
  %t2 = fadd float %a, %a
  %r = fadd float %t1, %t2
  ret float %r
}

define float @test4(float %a) {
; CHECK-LABEL: test4:
; CHECK:       # BB#0:
; CHECK-NEXT:    vmulss {{.*}}(%rip), %xmm0, %xmm0
; CHECK-NEXT:    retq
  %t1 = fadd float %a, %a
  %t2 = fmul float 4.0, %a
  %r = fadd float %t1, %t2
  ret float %r
}

define float @test5(float %a) {
; CHECK-LABEL: test5:
; CHECK:       # BB#0:
; CHECK-NEXT:    vmulss {{.*}}(%rip), %xmm0, %xmm0
; CHECK-NEXT:    retq
  %t1 = fadd float %a, %a
  %t2 = fmul float %a, 4.0
  %r = fadd float %t1, %t2
  ret float %r
}

define float @test6(float %a) {
; CHECK-LABEL: test6:
; CHECK:       # BB#0:
; CHECK-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %t1 = fmul float 2.0, %a
  %t2 = fadd float %a, %a
  %r = fsub float %t1, %t2
  ret float %r
}

define float @test7(float %a) {
; CHECK-LABEL: test7:
; CHECK:       # BB#0:
; CHECK-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %t1 = fmul float %a, 2.0
  %t2 = fadd float %a, %a
  %r = fsub float %t1, %t2
  ret float %r
}

define float @test8(float %a) {
; CHECK-LABEL: test8:
; CHECK:       # BB#0:
; CHECK-NEXT:    retq
  %t1 = fmul float %a, 0.0
  %t2 = fadd float %a, %t1
  ret float %t2
}

define float @test9(float %a) {
; CHECK-LABEL: test9:
; CHECK:       # BB#0:
; CHECK-NEXT:    retq
  %t1 = fmul float 0.0, %a
  %t2 = fadd float %t1, %a
  ret float %t2
}

define float @test10(float %a) {
; CHECK-LABEL: test10:
; CHECK:       # BB#0:
; CHECK-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %t1 = fsub float -0.0, %a
  %t2 = fadd float %a, %t1
  ret float %t2
}

define float @test11(float %a) {
; CHECK-LABEL: test11:
; CHECK:       # BB#0:
; CHECK-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %t1 = fsub float -0.0, %a
  %t2 = fadd float %a, %t1
  ret float %t2
}

; Verify that the first two adds are independent regardless of how the inputs are 
; commuted. The destination registers are used as source registers for the third add.

define float @reassociate_adds1(float %x0, float %x1, float %x2, float %x3) {
; CHECK-LABEL: reassociate_adds1:
; CHECK:       # BB#0:
; CHECK-NEXT:    vaddss %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vaddss %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vaddss %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %t0 = fadd float %x0, %x1
  %t1 = fadd float %t0, %x2
  %t2 = fadd float %t1, %x3
  ret float %t2
}

define float @reassociate_adds2(float %x0, float %x1, float %x2, float %x3) {
; CHECK-LABEL: reassociate_adds2:
; CHECK:       # BB#0:
; CHECK-NEXT:    vaddss %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vaddss %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vaddss %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %t0 = fadd float %x0, %x1
  %t1 = fadd float %x2, %t0
  %t2 = fadd float %t1, %x3
  ret float %t2
}

define float @reassociate_adds3(float %x0, float %x1, float %x2, float %x3) {
; CHECK-LABEL: reassociate_adds3:
; CHECK:       # BB#0:
; CHECK-NEXT:    vaddss %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vaddss %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vaddss %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %t0 = fadd float %x0, %x1
  %t1 = fadd float %t0, %x2
  %t2 = fadd float %x3, %t1
  ret float %t2
}

define float @reassociate_adds4(float %x0, float %x1, float %x2, float %x3) {
; CHECK-LABEL: reassociate_adds4:
; CHECK:       # BB#0:
; CHECK-NEXT:    vaddss %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vaddss %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vaddss %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %t0 = fadd float %x0, %x1
  %t1 = fadd float %x2, %t0
  %t2 = fadd float %x3, %t1
  ret float %t2
}

; Verify that we reassociate some of these ops. The optimal balanced tree of adds is not
; produced because that would cost more compile time.

define float @reassociate_adds5(float %x0, float %x1, float %x2, float %x3, float %x4, float %x5, float %x6, float %x7) {
; CHECK-LABEL: reassociate_adds5:
; CHECK:       # BB#0:
; CHECK-NEXT:    vaddss %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vaddss %xmm3, %xmm2, %xmm1
; CHECK-NEXT:    vaddss %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vaddss %xmm5, %xmm4, %xmm1
; CHECK-NEXT:    vaddss %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    vaddss %xmm7, %xmm6, %xmm1
; CHECK-NEXT:    vaddss %xmm1, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %t0 = fadd float %x0, %x1
  %t1 = fadd float %t0, %x2
  %t2 = fadd float %t1, %x3
  %t3 = fadd float %t2, %x4
  %t4 = fadd float %t3, %x5
  %t5 = fadd float %t4, %x6
  %t6 = fadd float %t5, %x7
  ret float %t6
}
