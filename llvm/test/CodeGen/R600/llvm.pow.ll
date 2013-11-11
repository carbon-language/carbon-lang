;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

;CHECK: LOG_IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;CHECK: MUL NON-IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], PS}}
;CHECK-NEXT: EXP_IEEE * T{{[0-9]+\.[XYZW], PV\.[XYZW]}}

define void @test(<4 x float> inreg %reg0) #0 {
   %r0 = extractelement <4 x float> %reg0, i32 0
   %r1 = extractelement <4 x float> %reg0, i32 1
   %r2 = call float @llvm.pow.f32( float %r0, float %r1)
   %vec = insertelement <4 x float> undef, float %r2, i32 0
   call void @llvm.R600.store.swizzle(<4 x float> %vec, i32 0, i32 0)
   ret void
}

declare float @llvm.pow.f32(float ,float ) readonly
declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)

attributes #0 = { "ShaderType"="0" }
