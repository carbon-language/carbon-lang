;RUN: llc < %s -march=r600 -mcpu=SI | FileCheck %s

;CHECK: V_MOV_B32_e32 VGPR1, -1431655765
;CHECK-NEXT: V_MUL_HI_U32 VGPR0, VGPR0, VGPR1, 0, 0, 0, 0, 0
;CHECK-NEXT: V_LSHRREV_B32_e32 VGPR0, 1, VGPR0

define void @test(i32 %p) {
   %i = udiv i32 %p, 3
   %r = bitcast i32 %i to float
   call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %r, float %r, float %r, float %r)
   ret void
}

declare <4 x float> @llvm.SI.sample.(i32, <4 x i32>, <8 x i32>, <4 x i32>, i32) readnone

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)
