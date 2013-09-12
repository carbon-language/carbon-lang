;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; CHECK: @main
; CHECK-NOT: MOV
define void @main() {
entry:
  %0 = call float @llvm.R600.load.input(i32 0)
  %1 = call float @fabs(float %0)
  %2 = fptoui float %1 to i32
  %3 = bitcast i32 %2 to float
  %4 = insertelement <4 x float> undef, float %3, i32 0
  call void @llvm.R600.store.swizzle(<4 x float> %4, i32 0, i32 0)
  ret void
}

declare float @llvm.R600.load.input(i32) readnone
declare float @fabs(float ) readnone
declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)