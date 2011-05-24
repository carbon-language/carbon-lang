; RUN: llc < %s -march=arm -mattr=+vfp2

define i32 @foo(float %scale, float %scale2) nounwind ssp {
entry:
  %scale.addr = alloca float, align 4
  %scale2.addr = alloca float, align 4
  store float %scale, float* %scale.addr, align 4
  store float %scale2, float* %scale2.addr, align 4
  %tmp = load float* %scale.addr, align 4
  %tmp1 = load float* %scale2.addr, align 4
  call void asm sideeffect "vmul.f32    q0, q0, ${0:y} \0A\09vmul.f32    q1, q1, ${0:y} \0A\09vmul.f32    q1, q0, ${1:y} \0A\09", "w,w,~{q0},~{q1}"(float %tmp, float %tmp1) nounwind, !srcloc !0
  ret i32 0
}

!0 = metadata !{i32 56, i32 89, i32 128, i32 168}
