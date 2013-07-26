; RUN: llc < %s -march=x86-64 -mattr=+sse41 -soft-float=0 | FileCheck %s --check-prefix=CHECK-HARD-FLOAT
; RUN: llc < %s -march=x86-64 -mattr=+sse41 -soft-float=1 | FileCheck %s --check-prefix=CHECK-SOFT-FLOAT

declare float @llvm.floor.f32(float)

; CHECK-SOFT-FLOAT: callq _floorf
; CHECK-HARD-FLOAT: vroundss $1, %xmm0, %xmm0, %xmm0
define float @myfloor(float %a) {
  %val = tail call float @llvm.floor.f32(float %a)
  ret float %val
}
