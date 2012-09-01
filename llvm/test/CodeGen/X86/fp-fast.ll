; RUN: llc -march=x86-64 -mattr=-fma4 -mtriple=x86_64-apple-darwin -enable-unsafe-fp-math < %s | FileCheck %s

; CHECK: test1
define float @test1(float %a) {
; CHECK-NOT: addss
; CHECK: mulss
; CHECK-NOT: addss
; CHECK: ret
  %t1 = fadd float %a, %a
  %r = fadd float %t1, %t1
  ret float %r
}

; CHECK: test2
define float @test2(float %a) {
; CHECK-NOT: addss
; CHECK: mulss
; CHECK-NOT: addss
; CHECK: ret
  %t1 = fmul float 4.0, %a
  %t2 = fadd float %a, %a
  %r = fadd float %t1, %t2
  ret float %r
}

; CHECK: test3
define float @test3(float %a) {
; CHECK-NOT: addss
; CHECK: xorps
; CHECK-NOT: addss
; CHECK: ret
  %t1 = fmul float 2.0, %a
  %t2 = fadd float %a, %a
  %r = fsub float %t1, %t2
  ret float %r
}

