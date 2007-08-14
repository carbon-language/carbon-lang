; RUN: llvm-as < %s | llc -march=x86-64 | grep LCPI | wc -l | grep 3
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep LCPI | wc -l | grep 3
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -stats |& | grep asm-printer | grep 13

declare float @qux(float %y)

define float @array(float %a) {
  %n = mul float %a, 9.0
  %m = call float @qux(float %n)
  %o = mul float %m, 9.0
  ret float %o
}
