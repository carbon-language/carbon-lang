; RUN: llc %s -O0 -march=sparc -mcpu=ut699 -o - | FileCheck %s

; CHECK-LABEL: test_fix_fsmuld_1
; CHECK:       fsmuld %f20, %f21, %f8

define double @test_fix_fsmuld_1() {
entry:
  %a = alloca float, align 4
  %b = alloca float, align 4
  store float 0x402ECCCCC0000000, float* %a, align 4
  store float 0x4022333340000000, float* %b, align 4
  %0 = load float, float* %b, align 4
  %1 = load float, float* %a, align 4
  %mul = tail call double asm sideeffect "fsmuld $0, $1, $2", "={f20},{f21},{f8}"(float* %a, float* %b)

  ret double %mul
}

; CHECK-LABEL: test_fix_fsmuld_2
; CHECK:       fsmuld %f20, %f21, %f8
define double @test_fix_fsmuld_2(float* %a, float* %b) {
entry:
  %mul = tail call double asm sideeffect "fsmuld $0, $1, $2", "={f20},{f21},{f8}"(float* %a, float* %b)

  ret double %mul
}
