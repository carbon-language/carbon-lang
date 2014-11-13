; RUN: llc -march=r600 -mcpu=SI < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: @test_fmax_legacy_uge_f32
; SI: v_max_legacy_f32_e32
; EG: MAX
define void @test_fmax_legacy_uge_f32(float addrspace(1)* %out, float %a, float %b) nounwind {
  %cmp = fcmp uge float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @test_fmax_legacy_oge_f32
; SI: v_max_legacy_f32_e32
; EG: MAX
define void @test_fmax_legacy_oge_f32(float addrspace(1)* %out, float %a, float %b) nounwind {
  %cmp = fcmp oge float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @test_fmax_legacy_ugt_f32
; SI: v_max_legacy_f32_e32
; EG: MAX
define void @test_fmax_legacy_ugt_f32(float addrspace(1)* %out, float %a, float %b) nounwind {
  %cmp = fcmp ugt float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @test_fmax_legacy_ogt_f32
; SI: v_max_legacy_f32_e32
; EG: MAX
define void @test_fmax_legacy_ogt_f32(float addrspace(1)* %out, float %a, float %b) nounwind {
  %cmp = fcmp ogt float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, float addrspace(1)* %out, align 4
  ret void
}
