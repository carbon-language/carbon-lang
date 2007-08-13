; RUN: llvm-as < %s | llc -march=x86-64 | grep LCPI | wc -l | grep 3
; RUN: llvm-as < %s | llc -march=x86 | grep LCPI | wc -l | grep 3

declare float @qux(float %y)

define float @array(float %a) {
  %n = mul float %a, 9.0
  %m = call float @qux(float %n)
  %o = mul float %m, 9.0
  ret float %o
}
