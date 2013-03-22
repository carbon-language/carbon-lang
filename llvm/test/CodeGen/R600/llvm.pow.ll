;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

;CHECK: LOG_IEEE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;CHECK-NEXT: MUL NON-IEEE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;CHECK-NEXT: EXP_IEEE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define void @test() {
   %r0 = call float @llvm.R600.load.input(i32 0)
   %r1 = call float @llvm.R600.load.input(i32 1)
   %r2 = call float @llvm.pow.f32( float %r0, float %r1)
   call void @llvm.AMDGPU.store.output(float %r2, i32 0)
   ret void
}

declare float @llvm.R600.load.input(i32) readnone

declare void @llvm.AMDGPU.store.output(float, i32)

declare float @llvm.pow.f32(float ,float ) readonly
