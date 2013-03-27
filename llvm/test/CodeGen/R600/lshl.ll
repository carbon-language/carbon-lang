;RUN: llc < %s -march=r600 -mcpu=SI | FileCheck %s

;CHECK: V_LSHL_B32_e64 VGPR0, VGPR0, 1, 0, 0, 0, 0

define void @test(i32 %p) {
   %i = mul i32 %p, 2
   %r = bitcast i32 %i to float
   call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %r, float %r, float %r, float %r)
   ret void
}

declare <4 x float> @llvm.SI.sample.(i32, <4 x i32>, <8 x i32>, <4 x i32>, i32) readnone

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)
