; RUN: llc < %s -mcpu=corei7 -march=x86-64 -mattr=+sse2 | FileCheck %s
; Verify that floating-point operations inside 'optnone' functions
; are not optimized even if unsafe-fp-math is set.

define float @foo(float %x) #0 {
entry:
  %add = fadd fast float %x, %x
  %add1 = fadd fast float %add, %x
  ret float %add1
}

; CHECK-LABEL: @foo
; CHECK-NOT: add
; CHECK: mul
; CHECK-NOT: add
; CHECK: ret

define float @fooWithOptnone(float %x) #1 {
entry:
  %add = fadd fast float %x, %x
  %add1 = fadd fast float %add, %x
  ret float %add1
}

; CHECK-LABEL: @fooWithOptnone
; CHECK-NOT: mul
; CHECK: add
; CHECK-NOT: mul
; CHECK: add
; CHECK-NOT: mul
; CHECK: ret


attributes #0 = { "unsafe-fp-math"="true" }
attributes #1 = { noinline optnone "unsafe-fp-math"="true" }
