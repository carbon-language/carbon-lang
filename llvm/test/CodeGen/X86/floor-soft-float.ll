; RUN: llc < %s -march=x86-64 -mattr=+sse4.1,-avx -soft-float=0 | FileCheck %s --check-prefix=CHECK-HARD-FLOAT
; RUN: llc < %s -march=x86-64 -mattr=+sse4.1,-avx -soft-float=1 | FileCheck %s --check-prefix=CHECK-SOFT-FLOAT

target triple = "x86_64-unknown-linux-gnu"

declare float @llvm.floor.f32(float)

; CHECK-SOFT-FLOAT: callq floorf
; CHECK-HARD-FLOAT: roundss $1, %xmm0, %xmm0
define float @myfloor(float %a) {
  %val = tail call float @llvm.floor.f32(float %a)
  ret float %val
}
