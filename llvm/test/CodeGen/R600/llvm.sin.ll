;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s -check-prefix=EG -check-prefix=FUNC
;RUN: llc < %s -march=r600 -mcpu=SI | FileCheck %s -check-prefix=SI -check-prefix=FUNC

;FUNC-LABEL: test
;EG: MULADD_IEEE *
;EG: FRACT *
;EG: ADD *
;EG: SIN * T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
;EG-NOT: SIN
;SI: V_MUL_F32
;SI: V_SIN_F32
;SI-NOT: V_SIN_F32

define void @test(float addrspace(1)* %out, float %x) #1 {
   %sin = call float @llvm.sin.f32(float %x)
   store float %sin, float addrspace(1)* %out
   ret void
}

;FUNC-LABEL: testv
;EG: SIN * T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
;EG: SIN * T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
;EG: SIN * T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
;EG: SIN * T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
;EG-NOT: SIN
;SI: V_SIN_F32
;SI: V_SIN_F32
;SI: V_SIN_F32
;SI: V_SIN_F32
;SI-NOT: V_SIN_F32

define void @testv(<4 x float> addrspace(1)* %out, <4 x float> %vx) #1 {
   %sin = call <4 x float> @llvm.sin.v4f32( <4 x float> %vx)
   store <4 x float> %sin, <4 x float> addrspace(1)* %out
   ret void
}

declare float @llvm.sin.f32(float) readnone
declare <4 x float> @llvm.sin.v4f32(<4 x float>) readnone

attributes #0 = { "ShaderType"="0" }
