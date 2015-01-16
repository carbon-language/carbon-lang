; RUN: llc -O3 -disable-peephole -mcpu=corei7-avx -mattr=+avx < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

define <4 x float> @stack_fold_cvtdq2ps(<4 x i32> %a) {
  ;CHECK-LABEL: stack_fold_cvtdq2ps
  ;CHECK:   vcvtdq2ps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15}"()
  %2 = sitofp <4 x i32> %a to <4 x float>
  ret <4 x float> %2
}

define <8 x float> @stack_fold_cvtdq2ps_ymm(<8 x i32> %a) {
  ;CHECK-LABEL: stack_fold_cvtdq2ps_ymm
  ;CHECK:   vcvtdq2ps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15}"()
  %2 = sitofp <8 x i32> %a to <8 x float>
  ret <8 x float> %2
}

define <4 x i32> @stack_fold_cvttpd2dq_ymm(<4 x double> %a) {
  ;CHECK-LABEL: stack_fold_cvttpd2dq_ymm
  ;CHECK:  vcvttpd2dqy {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15}"()
  %2 = fptosi <4 x double> %a to <4 x i32>
  ret <4 x i32> %2
}

define <2 x float> @stack_fold_cvtpd2ps(<2 x double> %a) {
  ;CHECK-LABEL: stack_fold_cvtpd2ps
  ;CHECK:   vcvtpd2psx {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15}"()
  %2 = fptrunc <2 x double> %a to <2 x float>
  ret <2 x float> %2
}

define <4 x float> @stack_fold_cvtpd2ps_ymm(<4 x double> %a) {
  ;CHECK-LABEL: stack_fold_cvtpd2ps_ymm
  ;CHECK:   vcvtpd2psy {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15}"()
  %2 = fptrunc <4 x double> %a to <4 x float>
  ret <4 x float> %2
}

define <4 x i32> @stack_fold_cvttps2dq(<4 x float> %a) {
  ;CHECK-LABEL: stack_fold_cvttps2dq
  ;CHECK:  vcvttps2dq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15}"()
  %2 = fptosi <4 x float> %a to <4 x i32>
  ret <4 x i32> %2
}

define <8 x i32> @stack_fold_cvttps2dq_ymm(<8 x float> %a) {
  ;CHECK-LABEL: stack_fold_cvttps2dq_ymm
  ;CHECK:  vcvttps2dq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15}"()
  %2 = fptosi <8 x float> %a to <8 x i32>
  ret <8 x i32> %2
}

define <2 x double> @stack_fold_vmulpd(<2 x double> %a, <2 x double> %b) {
  ;CHECK-LABEL: stack_fold_vmulpd
  ;CHECK:       vmulpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15}"()
  %2 = fmul <2 x double> %a, %b
  ret <2 x double> %2
}

define <4 x double> @stack_fold_vmulpd_ymm(<4 x double> %a, <4 x double> %b) {
  ;CHECK-LABEL: stack_fold_vmulpd_ymm
  ;CHECK:       vmulpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15}"()
  %2 = fmul <4 x double> %a, %b
  ret <4 x double> %2
}
