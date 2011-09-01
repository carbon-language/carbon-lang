; RUN: llc < %s -relocation-model=pic -disable-fp-elim -mcpu=cortex-a8 -pre-RA-sched=source | FileCheck %s
target triple = "thumbv7-apple-ios"
; <rdar://problem/10032939>
;
; The vector %v2 is built like this:
;
;   %vreg6:ssub_1<def> = VMOVSR %vreg0<kill>, pred:14, pred:%noreg, %vreg6<imp-def>; DPR_VFP2:%vreg6 GPR:%vreg0
;   %vreg6:ssub_0<def> = VLDRS <cp#0>, 0, pred:14, pred:%noreg; mem:LD4[ConstantPool] DPR_VFP2:%vreg6
;
; When %vreg6 spills, the VLDRS constant pool load cannot be rematerialized
; since it implicitly reads the ssub_1 sub-register.
;
; CHECK: f1
; CHECK: vmov    s1, r0
; CHECK: vldr.32 s0, LCPI
; The vector must be spilled:
; CHECK: vstr.64 d0,
; CHECK: asm clobber d0
; And reloaded after the asm:
; CHECK: vldr.64 [[D16:d[0-9]+]],
; CHECK: vstr.64 [[D16]], [r1]
define void @f1(float %x, <2 x float>* %p) {
  %v1 = insertelement <2 x float> undef, float %x, i32 1
  %v2 = insertelement <2 x float> %v1, float 0x400921FB60000000, i32 0
  %y = call double asm sideeffect "asm clobber $0", "=w,0,~{d1},~{d2},~{d3},~{d4},~{d5},~{d6},~{d7},~{d8},~{d9},~{d10},~{d11},~{d12},~{d13},~{d14},~{d15},~{d16},~{d17},~{d18},~{d19},~{d20},~{d21},~{d22},~{d23},~{d24},~{d25},~{d26},~{d27},~{d28},~{d29},~{d30},~{d31}"(<2 x float> %v2) nounwind
  store <2 x float> %v2, <2 x float>* %p, align 8
  ret void
}
