; RUN: llc -march=r600 -mcpu=SI < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: @test_fmin_legacy_f32
; EG: MIN *
; SI: v_min_legacy_f32_e32
define void @test_fmin_legacy_f32(<4 x float> addrspace(1)* %out, <4 x float> inreg %reg0) nounwind {
   %r0 = extractelement <4 x float> %reg0, i32 0
   %r1 = extractelement <4 x float> %reg0, i32 1
   %r2 = fcmp uge float %r0, %r1
   %r3 = select i1 %r2, float %r1, float %r0
   %vec = insertelement <4 x float> undef, float %r3, i32 0
   store <4 x float> %vec, <4 x float> addrspace(1)* %out, align 16
   ret void
}

; FUNC-LABEL: @test_fmin_legacy_ule_f32
; SI: v_min_legacy_f32_e32
define void @test_fmin_legacy_ule_f32(float addrspace(1)* %out, float %a, float %b) nounwind {
  %cmp = fcmp ule float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @test_fmin_legacy_ole_f32
; SI: v_min_legacy_f32_e32
define void @test_fmin_legacy_ole_f32(float addrspace(1)* %out, float %a, float %b) nounwind {
  %cmp = fcmp ole float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @test_fmin_legacy_olt_f32
; SI: v_min_legacy_f32_e32
define void @test_fmin_legacy_olt_f32(float addrspace(1)* %out, float %a, float %b) nounwind {
  %cmp = fcmp olt float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @test_fmin_legacy_ult_f32
; SI: v_min_legacy_f32_e32
define void @test_fmin_legacy_ult_f32(float addrspace(1)* %out, float %a, float %b) nounwind {
  %cmp = fcmp ult float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, float addrspace(1)* %out, align 4
  ret void
}
