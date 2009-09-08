; RUN: llc < %s -march=x86-64 | grep LCPI | count 3
; RUN: llc < %s -march=x86-64 -stats  -info-output-file - | grep asm-printer | grep 6
; RUN: llc < %s -march=x86 -mattr=+sse2 | grep LCPI | count 3
; RUN: llc < %s -march=x86 -mattr=+sse2 -stats  -info-output-file - | grep asm-printer | grep 12

declare float @qux(float %y)

define float @array(float %a) nounwind {
  %n = fmul float %a, 9.0
  %m = call float @qux(float %n)
  %o = fmul float %m, 9.0
  ret float %o
}
