; RUN: llc < %s -march=nvptx -mcpu=sm_20 -nvptx-prec-divf32=0 | FileCheck %s

define float @foo(float %a) {
; CHECK: div.approx.f32
  %div = fdiv float %a, 13.0
  ret float %div
}

