; RUN: llc -O3 -disable-peephole -mcpu=corei7-avx -mattr=+avx < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

define <2 x double> @stack_fold_addpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_addpd
  ;CHECK:       vaddpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fadd <2 x double> %a0, %a1
  ret <2 x double> %2
}

define <4 x double> @stack_fold_addpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_addpd_ymm
  ;CHECK:       vaddpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fadd <4 x double> %a0, %a1
  ret <4 x double> %2
}

define <4 x float> @stack_fold_addps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_addps
  ;CHECK:       vaddps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fadd <4 x float> %a0, %a1
  ret <4 x float> %2
}

define <8 x float> @stack_fold_addps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_addps_ymm
  ;CHECK:       vaddps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fadd <8 x float> %a0, %a1
  ret <8 x float> %2
}

define double @stack_fold_addsd(double %a0, double %a1) {
  ;CHECK-LABEL: stack_fold_addsd
  ;CHECK:       vaddsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fadd double %a0, %a1
  ret double %2
}

; TODO stack_fold_addsd_int
declare <2 x double> @llvm.x86.sse2.add.sd(<2 x double>, <2 x double>) nounwind readnone

define float @stack_fold_addss(float %a0, float %a1) {
  ;CHECK-LABEL: stack_fold_addss
  ;CHECK:       vaddss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fadd float %a0, %a1
  ret float %2
}

; TODO stack_fold_addss_int
declare <4 x float> @llvm.x86.sse.add.ss(<4 x float>, <4 x float>) nounwind readnone

define <2 x double> @stack_fold_addsubpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_addsubpd
  ;CHECK:       vaddsubpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse3.addsub.pd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse3.addsub.pd(<2 x double>, <2 x double>) nounwind readnone

define <4 x double> @stack_fold_addsubpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_addsubpd_ymm
  ;CHECK:       vaddsubpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx.addsub.pd.256(<4 x double> %a0, <4 x double> %a1)
  ret <4 x double> %2
}
declare <4 x double> @llvm.x86.avx.addsub.pd.256(<4 x double>, <4 x double>) nounwind readnone

define <4 x float> @stack_fold_addsubps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_addsubps
  ;CHECK:       vaddsubps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse3.addsub.ps(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse3.addsub.ps(<4 x float>, <4 x float>) nounwind readnone

define <8 x float> @stack_fold_addsubps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_addsubps_ymm
  ;CHECK:       vaddsubps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.addsub.ps.256(<8 x float> %a0, <8 x float> %a1)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx.addsub.ps.256(<8 x float>, <8 x float>) nounwind readnone

define <2 x double> @stack_fold_andnpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_andnpd
  ;CHECK:       vandnpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <2 x double> %a0 to <2 x i64>
  %3 = bitcast <2 x double> %a1 to <2 x i64>
  %4 = xor <2 x i64> %2, <i64 -1, i64 -1>
  %5 = and <2 x i64> %4, %3
  %6 = bitcast <2 x i64> %5 to <2 x double>
  ; fadd forces execution domain
  %7 = fadd <2 x double> %6, <double 0x0, double 0x0>
  ret <2 x double> %7
}

define <4 x double> @stack_fold_andnpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_andnpd_ymm
  ;CHECK:       vandnpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <4 x double> %a0 to <4 x i64>
  %3 = bitcast <4 x double> %a1 to <4 x i64>
  %4 = xor <4 x i64> %2, <i64 -1, i64 -1, i64 -1, i64 -1>
  %5 = and <4 x i64> %4, %3
  %6 = bitcast <4 x i64> %5 to <4 x double>
  ; fadd forces execution domain
  %7 = fadd <4 x double> %6, <double 0x0, double 0x0, double 0x0, double 0x0>
  ret <4 x double> %7
}

define <4 x float> @stack_fold_andnps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_andnps
  ;CHECK:       vandnps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <4 x float> %a0 to <2 x i64>
  %3 = bitcast <4 x float> %a1 to <2 x i64>
  %4 = xor <2 x i64> %2, <i64 -1, i64 -1>
  %5 = and <2 x i64> %4, %3
  %6 = bitcast <2 x i64> %5 to <4 x float>
  ret <4 x float> %6
}

define <8 x float> @stack_fold_andnps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_andnps_ymm
  ;CHECK:       vandnps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <8 x float> %a0 to <4 x i64>
  %3 = bitcast <8 x float> %a1 to <4 x i64>
  %4 = xor <4 x i64> %2, <i64 -1, i64 -1, i64 -1, i64 -1>
  %5 = and <4 x i64> %4, %3
  %6 = bitcast <4 x i64> %5 to <8 x float>
  ret <8 x float> %6
}

define <2 x double> @stack_fold_andpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_andpd
  ;CHECK:       vandpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <2 x double> %a0 to <2 x i64>
  %3 = bitcast <2 x double> %a1 to <2 x i64>
  %4 = and <2 x i64> %2, %3
  %5 = bitcast <2 x i64> %4 to <2 x double>
  ; fadd forces execution domain
  %6 = fadd <2 x double> %5, <double 0x0, double 0x0>
  ret <2 x double> %6
}

define <4 x double> @stack_fold_andpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_andpd_ymm
  ;CHECK:       vandpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <4 x double> %a0 to <4 x i64>
  %3 = bitcast <4 x double> %a1 to <4 x i64>
  %4 = and <4 x i64> %2, %3
  %5 = bitcast <4 x i64> %4 to <4 x double>
  ; fadd forces execution domain
  %6 = fadd <4 x double> %5, <double 0x0, double 0x0, double 0x0, double 0x0>
  ret <4 x double> %6
}

define <4 x float> @stack_fold_andps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_andps
  ;CHECK:       vandps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <4 x float> %a0 to <2 x i64>
  %3 = bitcast <4 x float> %a1 to <2 x i64>
  %4 = and <2 x i64> %2, %3
  %5 = bitcast <2 x i64> %4 to <4 x float>
  ret <4 x float> %5
}

define <8 x float> @stack_fold_andps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_andps_ymm
  ;CHECK:       vandps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <8 x float> %a0 to <4 x i64>
  %3 = bitcast <8 x float> %a1 to <4 x i64>
  %4 = and <4 x i64> %2, %3
  %5 = bitcast <4 x i64> %4 to <8 x float>
  ret <8 x float> %5
}

define <2 x double> @stack_fold_blendpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_blendpd
  ;CHECK:       vblendpd $2, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = select <2 x i1> <i1 1, i1 0>, <2 x double> %a0, <2 x double> %a1
  ret <2 x double> %2
}

define <4 x double> @stack_fold_blendpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_blendpd_ymm
  ;CHECK:       vblendpd $6, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = select <4 x i1> <i1 1, i1 0, i1 0, i1 1>, <4 x double> %a0, <4 x double> %a1
  ret <4 x double> %2
}

define <4 x float> @stack_fold_blendps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_blendps
  ;CHECK:       vblendps $6, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = select <4 x i1> <i1 1, i1 0, i1 0, i1 1>, <4 x float> %a0, <4 x float> %a1
  ret <4 x float> %2
}

define <8 x float> @stack_fold_blendps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_blendps_ymm
  ;CHECK:       vblendps $102, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = select <8 x i1> <i1 1, i1 0, i1 0, i1 1, i1 1, i1 0, i1 0, i1 1>, <8 x float> %a0, <8 x float> %a1
  ret <8 x float> %2
}

define <2 x double> @stack_fold_blendvpd(<2 x double> %a0, <2 x double> %a1, <2 x double> %c) {
  ;CHECK-LABEL: stack_fold_blendvpd
  ;CHECK:       vblendvpd {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse41.blendvpd(<2 x double> %a1, <2 x double> %c, <2 x double> %a0)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse41.blendvpd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone

define <4 x double> @stack_fold_blendvpd_ymm(<4 x double> %a0, <4 x double> %a1, <4 x double> %c) {
  ;CHECK-LABEL: stack_fold_blendvpd_ymm
  ;CHECK:       vblendvpd {{%ymm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx.blendv.pd.256(<4 x double> %a1, <4 x double> %c, <4 x double> %a0)
  ret <4 x double> %2
}
declare <4 x double> @llvm.x86.avx.blendv.pd.256(<4 x double>, <4 x double>, <4 x double>) nounwind readnone

define <4 x float> @stack_fold_blendvps(<4 x float> %a0, <4 x float> %a1, <4 x float> %c) {
  ;CHECK-LABEL: stack_fold_blendvps
  ;CHECK:       vblendvps {{%xmm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %a1, <4 x float> %c, <4 x float> %a0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse41.blendvps(<4 x float>, <4 x float>, <4 x float>) nounwind readnone

define <8 x float> @stack_fold_blendvps_ymm(<8 x float> %a0, <8 x float> %a1, <8 x float> %c) {
  ;CHECK-LABEL: stack_fold_blendvps_ymm
  ;CHECK:       vblendvps {{%ymm[0-9][0-9]*}}, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float> %a1, <8 x float> %c, <8 x float> %a0)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float>, <8 x float>, <8 x float>) nounwind readnone

define <2 x double> @stack_fold_cmppd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_cmppd
  ;CHECK:       vcmpeqpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double> %a0, <2 x double> %a1, i8 0)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double>, <2 x double>, i8) nounwind readnone

define <4 x double> @stack_fold_cmppd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_cmppd_ymm
  ;CHECK:       vcmpeqpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx.cmp.pd.256(<4 x double> %a0, <4 x double> %a1, i8 0)
  ret <4 x double> %2
}
declare <4 x double> @llvm.x86.avx.cmp.pd.256(<4 x double>, <4 x double>, i8) nounwind readnone

define <4 x float> @stack_fold_cmpps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_cmpps
  ;CHECK:       vcmpeqps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.cmp.ps(<4 x float> %a0, <4 x float> %a1, i8 0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.cmp.ps(<4 x float>, <4 x float>, i8) nounwind readnone

define <8 x float> @stack_fold_cmpps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_cmpps_ymm
  ;CHECK:       vcmpeqps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float> %a0, <8 x float> %a1, i8 0)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx.cmp.ps.256(<8 x float>, <8 x float>, i8) nounwind readnone

define i32 @stack_fold_cmpsd(double %a0, double %a1) {
  ;CHECK-LABEL: stack_fold_cmpsd
  ;CHECK:       vcmpeqsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp oeq double %a0, %a1
  %3 = zext i1 %2 to i32
  ret i32 %3
}

define <2 x double> @stack_fold_cmpsd_int(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_cmpsd_int
  ;CHECK:       vcmpeqsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %a0, <2 x double> %a1, i8 0)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double>, <2 x double>, i8) nounwind readnone

define i32 @stack_fold_cmpss(float %a0, float %a1) {
  ;CHECK-LABEL: stack_fold_cmpss
  ;CHECK:       vcmpeqss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp oeq float %a0, %a1
  %3 = zext i1 %2 to i32
  ret i32 %3
}

define <4 x float> @stack_fold_cmpss_int(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_cmpss_int
  ;CHECK:       vcmpeqss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.cmp.ss(<4 x float> %a0, <4 x float> %a1, i8 0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.cmp.ss(<4 x float>, <4 x float>, i8) nounwind readnone

; TODO stack_fold_comisd

define i32 @stack_fold_comisd_int(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_comisd_int
  ;CHECK:       vcomisd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i32 @llvm.x86.sse2.comieq.sd(<2 x double> %a0, <2 x double> %a1)
  ret i32 %2
}
declare i32 @llvm.x86.sse2.comieq.sd(<2 x double>, <2 x double>) nounwind readnone

; TODO stack_fold_comiss

define i32 @stack_fold_comiss_int(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_comiss_int
  ;CHECK:       vcomiss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i32 @llvm.x86.sse.comieq.ss(<4 x float> %a0, <4 x float> %a1)
  ret i32 %2
}
declare i32 @llvm.x86.sse.comieq.ss(<4 x float>, <4 x float>) nounwind readnone

; TODO stack_fold_cvtdq2pd
declare <2 x double> @llvm.x86.sse2.cvtdq2pd(<4 x i32>) nounwind readnone

; TODO stack_fold_cvtdq2pd_ymm
declare <4 x double> @llvm.x86.avx.cvtdq2.pd.256(<4 x i32>) nounwind readnone

define <4 x float> @stack_fold_cvtdq2ps(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_cvtdq2ps
  ;CHECK:   vcvtdq2ps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = sitofp <4 x i32> %a0 to <4 x float>
  ret <4 x float> %2
}

define <8 x float> @stack_fold_cvtdq2ps_ymm(<8 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_cvtdq2ps_ymm
  ;CHECK:   vcvtdq2ps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = sitofp <8 x i32> %a0 to <8 x float>
  ret <8 x float> %2
}

define <4 x i32> @stack_fold_cvtpd2dq(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvtpd2dq
  ;CHECK:   vcvtpd2dqx {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse2.cvtpd2dq(<2 x double> %a0)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sse2.cvtpd2dq(<2 x double>) nounwind readnone

define <4 x i32> @stack_fold_cvtpd2dq_ymm(<4 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvtpd2dq_ymm
  ;CHECK:   vcvtpd2dqy {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.avx.cvt.pd2dq.256(<4 x double> %a0)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.avx.cvt.pd2dq.256(<4 x double>) nounwind readnone

define <2 x float> @stack_fold_cvtpd2ps(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvtpd2ps
  ;CHECK:   vcvtpd2psx {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fptrunc <2 x double> %a0 to <2 x float>
  ret <2 x float> %2
}

define <4 x float> @stack_fold_cvtpd2ps_ymm(<4 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvtpd2ps_ymm
  ;CHECK:   vcvtpd2psy {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fptrunc <4 x double> %a0 to <4 x float>
  ret <4 x float> %2
}

; TODO stack_fold_cvtph2ps
declare <4 x float> @llvm.x86.vcvtph2ps.128(<8 x i16>) nounwind readonly

; TODO stack_fold_cvtph2ps_ymm
declare <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16>) nounwind readonly

define <4 x i32> @stack_fold_cvtps2dq(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvtps2dq
  ;CHECK:  vcvtps2dq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse2.cvtps2dq(<4 x float> %a0)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sse2.cvtps2dq(<4 x float>) nounwind readnone

define <8 x i32> @stack_fold_cvtps2dq_ymm(<8 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvtps2dq_ymm
  ;CHECK:  vcvtps2dq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i32> @llvm.x86.avx.cvt.ps2dq.256(<8 x float> %a0)
  ret <8 x i32> %2
}
declare <8 x i32> @llvm.x86.avx.cvt.ps2dq.256(<8 x float>) nounwind readnone

; TODO stack_fold_cvtps2pd
; TODO stack_fold_cvtps2pd_ymm

; TODO stack_fold_cvtps2ph
declare <8 x i16> @llvm.x86.vcvtps2ph.128(<4 x float>, i32) nounwind readonly

; TODO stack_fold_cvtps2ph_ymm
declare <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float>, i32) nounwind readonly

; TODO stack_fold_cvtsd2si (+ int)
declare i32 @llvm.x86.sse2.cvtsd2si(<2 x double>) nounwind readnone

; TODO stack_fold_cvtsd2si64 (+ int)
declare i64 @llvm.x86.sse2.cvtsd2si64(<2 x double>) nounwind readnone

; TODO stack_fold_cvtsd2ss

define <4 x float> @stack_fold_cvtsd2ss_int(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvtsd2ss_int
  ;CHECK:  cvtsd2ss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse2.cvtsd2ss(<4 x float> <float 0x0, float 0x0, float 0x0, float 0x0>, <2 x double> %a0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse2.cvtsd2ss(<4 x float>, <2 x double>) nounwind readnone

; TODO stack_fold_cvtsi2sd (+ int)
declare <2 x double> @llvm.x86.sse2.cvtsi2sd(<2 x double>, i32) nounwind readnone

; TODO stack_fold_cvtsi642sd (+ int)
declare <2 x double> @llvm.x86.sse2.cvtsi642sd(<2 x double>, i64) nounwind readnone

; TODO stack_fold_cvtsi2ss (+ int)
declare <4 x float> @llvm.x86.sse.cvtsi2ss(<4 x float>, i32) nounwind readnone

; TODO stack_fold_cvtsi642ss (+ int)
declare <4 x float> @llvm.x86.sse.cvtsi642ss(<4 x float>, i64) nounwind readnone

; TODO stack_fold_cvtss2sd

define <2 x double> @stack_fold_cvtss2sd_int(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvtss2sd_int
  ;CHECK:  cvtss2sd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.cvtss2sd(<2 x double> <double 0x0, double 0x0>, <4 x float> %a0)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.cvtss2sd(<2 x double>, <4 x float>) nounwind readnone

; TODO stack_fold_cvtss2si (+ int)
declare i32 @llvm.x86.sse.cvtss2si(<4 x float>) nounwind readnone

; TODO stack_fold_cvtss2si64 (+ int)
declare i64 @llvm.x86.sse.cvtss2si64(<4 x float>) nounwind readnone

; TODO stack_fold_cvttpd2dq

define <4 x i32> @stack_fold_cvttpd2dq_ymm(<4 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvttpd2dq_ymm
  ;CHECK:  vcvttpd2dqy {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fptosi <4 x double> %a0 to <4 x i32>
  ret <4 x i32> %2
}

define <4 x i32> @stack_fold_cvttps2dq(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvttps2dq
  ;CHECK:  vcvttps2dq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fptosi <4 x float> %a0 to <4 x i32>
  ret <4 x i32> %2
}

define <8 x i32> @stack_fold_cvttps2dq_ymm(<8 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvttps2dq_ymm
  ;CHECK:  vcvttps2dq {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fptosi <8 x float> %a0 to <8 x i32>
  ret <8 x i32> %2
}

; TODO stack_fold_cvttsd2si (+ int)
declare i32 @llvm.x86.sse2.cvttsd2si(<2 x double>) nounwind readnone

; TODO stack_fold_cvttsd2si64 (+ int)
declare i64 @llvm.x86.sse2.cvttsd2si64(<2 x double>) nounwind readnone

; TODO stack_fold_cvttss2si (+ int)
declare i32 @llvm.x86.sse.cvttss2si(<4 x float>) nounwind readnone

; TODO stack_fold_cvttss2si64 (+ int)
declare i64 @llvm.x86.sse.cvttss2si64(<4 x float>) nounwind readnone

define <2 x double> @stack_fold_divpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_divpd
  ;CHECK:       vdivpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fdiv <2 x double> %a0, %a1
  ret <2 x double> %2
}

define <4 x double> @stack_fold_divpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_divpd_ymm
  ;CHECK:       vdivpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fdiv <4 x double> %a0, %a1
  ret <4 x double> %2
}

define <4 x float> @stack_fold_divps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_divps
  ;CHECK:       vdivps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fdiv <4 x float> %a0, %a1
  ret <4 x float> %2
}

define <8 x float> @stack_fold_divps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_divps_ymm
  ;CHECK:       vdivps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fdiv <8 x float> %a0, %a1
  ret <8 x float> %2
}

define double @stack_fold_divsd(double %a0, double %a1) {
  ;CHECK-LABEL: stack_fold_divsd
  ;CHECK:       vdivsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fdiv double %a0, %a1
  ret double %2
}

; TODO stack_fold_divsd_int
declare <2 x double> @llvm.x86.sse2.div.sd(<2 x double>, <2 x double>) nounwind readnone

define float @stack_fold_divss(float %a0, float %a1) {
  ;CHECK-LABEL: stack_fold_divss
  ;CHECK:       vdivss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fdiv float %a0, %a1
  ret float %2
}

; TODO stack_fold_divss_int
declare <4 x float> @llvm.x86.sse.div.ss(<4 x float>, <4 x float>) nounwind readnone

; TODO stack_fold_dppd
; TODO stack_fold_dppd_ymm
; TODO stack_fold_dpps
; TODO stack_fold_dpps_ymm

; TODO stack_fold_extractf128
; TODO stack_fold_extractps

define <2 x double> @stack_fold_haddpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_haddpd
  ;CHECK:       vhaddpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse3.hadd.pd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse3.hadd.pd(<2 x double>, <2 x double>) nounwind readnone

define <4 x double> @stack_fold_haddpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_haddpd_ymm
  ;CHECK:       vhaddpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double> %a0, <4 x double> %a1)
  ret <4 x double> %2
}
declare <4 x double> @llvm.x86.avx.hadd.pd.256(<4 x double>, <4 x double>) nounwind readnone

define <4 x float> @stack_fold_haddps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_haddps
  ;CHECK:       vhaddps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float>, <4 x float>) nounwind readnone

define <8 x float> @stack_fold_haddps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_haddps_ymm
  ;CHECK:       vhaddps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float> %a0, <8 x float> %a1)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx.hadd.ps.256(<8 x float>, <8 x float>) nounwind readnone

define <2 x double> @stack_fold_hsubpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_hsubpd
  ;CHECK:       vhsubpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse3.hsub.pd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse3.hsub.pd(<2 x double>, <2 x double>) nounwind readnone

define <4 x double> @stack_fold_hsubpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_hsubpd_ymm
  ;CHECK:       vhsubpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx.hsub.pd.256(<4 x double> %a0, <4 x double> %a1)
  ret <4 x double> %2
}
declare <4 x double> @llvm.x86.avx.hsub.pd.256(<4 x double>, <4 x double>) nounwind readnone

define <4 x float> @stack_fold_hsubps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_hsubps
  ;CHECK:       vhsubps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse3.hsub.ps(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse3.hsub.ps(<4 x float>, <4 x float>) nounwind readnone

define <8 x float> @stack_fold_hsubps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_hsubps_ymm
  ;CHECK:       vhsubps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.hsub.ps.256(<8 x float> %a0, <8 x float> %a1)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx.hsub.ps.256(<8 x float>, <8 x float>) nounwind readnone

; TODO stack_fold_insertf128
; TODO stack_fold_insertps

define <2 x double> @stack_fold_maxpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_maxpd
  ;CHECK:       vmaxpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.max.pd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.max.pd(<2 x double>, <2 x double>) nounwind readnone

define <4 x double> @stack_fold_maxpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_maxpd_ymm
  ;CHECK:       vmaxpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx.max.pd.256(<4 x double> %a0, <4 x double> %a1)
  ret <4 x double> %2
}
declare <4 x double> @llvm.x86.avx.max.pd.256(<4 x double>, <4 x double>) nounwind readnone

define <4 x float> @stack_fold_maxps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_maxps
  ;CHECK:       vmaxps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.max.ps(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.max.ps(<4 x float>, <4 x float>) nounwind readnone

define <8 x float> @stack_fold_maxps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_maxps_ymm
  ;CHECK:       vmaxps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.max.ps.256(<8 x float> %a0, <8 x float> %a1)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx.max.ps.256(<8 x float>, <8 x float>) nounwind readnone

; TODO stack_fold_maxsd (+ int)
declare <2 x double> @llvm.x86.sse2.max.sd(<2 x double>, <2 x double>) nounwind readnone

; TODO stack_fold_maxss (+ int)
declare <4 x float> @llvm.x86.sse.max.ss(<4 x float>, <4 x float>) nounwind readnone

define <2 x double> @stack_fold_minpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_minpd
  ;CHECK:       vminpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.min.pd(<2 x double> %a0, <2 x double> %a1)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.min.pd(<2 x double>, <2 x double>) nounwind readnone

define <4 x double> @stack_fold_minpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_minpd_ymm
  ;CHECK:       vminpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx.min.pd.256(<4 x double> %a0, <4 x double> %a1)
  ret <4 x double> %2
}
declare <4 x double> @llvm.x86.avx.min.pd.256(<4 x double>, <4 x double>) nounwind readnone

define <4 x float> @stack_fold_minps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_minps
  ;CHECK:       vminps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.min.ps(<4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.min.ps(<4 x float>, <4 x float>) nounwind readnone

define <8 x float> @stack_fold_minps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_minps_ymm
  ;CHECK:       vminps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.min.ps.256(<8 x float> %a0, <8 x float> %a1)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx.min.ps.256(<8 x float>, <8 x float>) nounwind readnone

; TODO stack_fold_minsd (+ int)
declare <2 x double> @llvm.x86.sse2.min.sd(<2 x double>, <2 x double>) nounwind readnone

; TODO stack_fold_minss (+ int)
declare <4 x float> @llvm.x86.sse.min.ss(<4 x float>, <4 x float>) nounwind readnone

; TODO stack_fold_movd (load / store)
; TODO stack_fold_movq (load / store)

; TODO stack_fold_movddup
; TODO stack_fold_movddup_ymm

; TODO stack_fold_movhpd (load / store)
; TODO stack_fold_movhps (load / store)

; TODO stack_fold_movlpd (load / store)
; TODO stack_fold_movlps (load / store)

; TODO stack_fold_movsd (load / store)
; TODO stack_fold_movss (load / store)

define <4 x float> @stack_fold_movshdup(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_movshdup
  ;CHECK:   vmovshdup {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> undef, <4 x i32> <i32 1, i32 1, i32 3, i32 3>
  ret <4 x float> %2
}

; TODO stack_fold_movshdup_ymm

define <4 x float> @stack_fold_movsldup(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_movsldup
  ;CHECK:   vmovsldup {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> undef, <4 x i32> <i32 0, i32 0, i32 2, i32 2>
  ret <4 x float> %2
}

; TODO stack_fold_movshdup_ymm

define <2 x double> @stack_fold_mulpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_mulpd
  ;CHECK:       vmulpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fmul <2 x double> %a0, %a1
  ret <2 x double> %2
}

define <4 x double> @stack_fold_mulpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_mulpd_ymm
  ;CHECK:       vmulpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fmul <4 x double> %a0, %a1
  ret <4 x double> %2
}

define <4 x float> @stack_fold_mulps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_mulps
  ;CHECK:       vmulps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fmul <4 x float> %a0, %a1
  ret <4 x float> %2
}

define <8 x float> @stack_fold_mulps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_mulps_ymm
  ;CHECK:       vmulps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fmul <8 x float> %a0, %a1
  ret <8 x float> %2
}

define double @stack_fold_mulsd(double %a0, double %a1) {
  ;CHECK-LABEL: stack_fold_mulsd
  ;CHECK:       vmulsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fmul double %a0, %a1
  ret double %2
}

; TODO stack_fold_mulsd_int
declare <2 x double> @llvm.x86.sse2.mul.sd(<2 x double>, <2 x double>) nounwind readnone

define float @stack_fold_mulss(float %a0, float %a1) {
  ;CHECK-LABEL: stack_fold_mulss
  ;CHECK:       vmulss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fmul float %a0, %a1
  ret float %2
}

; TODO stack_fold_mulss_int
declare <4 x float> @llvm.x86.sse.mul.ss(<4 x float>, <4 x float>) nounwind readnone

define <2 x double> @stack_fold_orpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_orpd
  ;CHECK:       vorpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <2 x double> %a0 to <2 x i64>
  %3 = bitcast <2 x double> %a1 to <2 x i64>
  %4 = or <2 x i64> %2, %3
  %5 = bitcast <2 x i64> %4 to <2 x double>
  ; fadd forces execution domain
  %6 = fadd <2 x double> %5, <double 0x0, double 0x0>
  ret <2 x double> %6
}

define <4 x double> @stack_fold_orpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_orpd_ymm
  ;CHECK:       vorpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <4 x double> %a0 to <4 x i64>
  %3 = bitcast <4 x double> %a1 to <4 x i64>
  %4 = or <4 x i64> %2, %3
  %5 = bitcast <4 x i64> %4 to <4 x double>
  ; fadd forces execution domain
  %6 = fadd <4 x double> %5, <double 0x0, double 0x0, double 0x0, double 0x0>
  ret <4 x double> %6
}

define <4 x float> @stack_fold_orps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_orps
  ;CHECK:       vorps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <4 x float> %a0 to <2 x i64>
  %3 = bitcast <4 x float> %a1 to <2 x i64>
  %4 = or <2 x i64> %2, %3
  %5 = bitcast <2 x i64> %4 to <4 x float>
  ret <4 x float> %5
}

define <8 x float> @stack_fold_orps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_orps_ymm
  ;CHECK:       vorps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <8 x float> %a0 to <4 x i64>
  %3 = bitcast <8 x float> %a1 to <4 x i64>
  %4 = or <4 x i64> %2, %3
  %5 = bitcast <4 x i64> %4 to <8 x float>
  ret <8 x float> %5
}

define <8 x float> @stack_fold_perm2f128(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_perm2f128
  ;CHECK:   vperm2f128 $33, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <8 x float> %a0, <8 x float> %a1, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11>
  ret <8 x float> %2
}

define <2 x double> @stack_fold_permilpd(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_permilpd
  ;CHECK:   vpermilpd $1, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <2 x double> %a0, <2 x double> undef, <2 x i32> <i32 1, i32 0>
  ret <2 x double> %2
}

define <4 x double> @stack_fold_permilpd_ymm(<4 x double> %a0) {
  ;CHECK-LABEL: stack_fold_permilpd_ymm
  ;CHECK:   vpermilpd $5, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x double> %a0, <4 x double> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  ret <4 x double> %2
}

define <4 x float> @stack_fold_permilps(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_permilps
  ;CHECK:   vpermilps $27, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x float> %2
}

define <8 x float> @stack_fold_permilps_ymm(<8 x float> %a0) {
  ;CHECK-LABEL: stack_fold_permilps_ymm
  ;CHECK:   vpermilps $27, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <8 x float> %a0, <8 x float> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  ret <8 x float> %2
}

; TODO stack_fold_rcpps

define <4 x float> @stack_fold_rcpps_int(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_rcpps_int
  ;CHECK:       vrcpps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.rcp.ps(<4 x float> %a0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.rcp.ps(<4 x float>) nounwind readnone

; TODO stack_fold_rcpps_ymm

define <8 x float> @stack_fold_rcpps_ymm_int(<8 x float> %a0) {
  ;CHECK-LABEL: stack_fold_rcpps_ymm_int
  ;CHECK:       vrcpps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.rcp.ps.256(<8 x float> %a0)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx.rcp.ps.256(<8 x float>) nounwind readnone

; TODO stack_fold_rcpss
; TODO stack_fold_rcpss_int

; TODO stack_fold_roundpd
; TODO stack_fold_roundps
; TODO stack_fold_roundsd (+ int)
; TODO stack_fold_roundss (+ int)

; TODO stack_fold_rsqrtps

define <4 x float> @stack_fold_rsqrtps_int(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_rsqrtps_int
  ;CHECK:       vrsqrtps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float> %a0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float>) nounwind readnone

; TODO stack_fold_rsqrtps_ymm
; TODO stack_fold_rsqrtps_ymm_int
; TODO stack_fold_rsqrtss
; TODO stack_fold_rsqrtss_int

define <2 x double> @stack_fold_shufpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_shufpd
  ;CHECK:       vshufpd $1, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <2 x double> %a0, <2 x double> %a1, <2 x i32> <i32 1, i32 2>
  ret <2 x double> %2
}

define <4 x double> @stack_fold_shufpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_shufpd_ymm
  ;CHECK:       vshufpd $5, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x double> %a0, <4 x double> %a1, <4 x i32> <i32 1, i32 4, i32 3, i32 6>
  ret <4 x double> %2
}

define <4 x float> @stack_fold_shufps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_shufps
  ;CHECK:       vshufps $-56, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> %a1, <4 x i32> <i32 0, i32 2, i32 4, i32 7>
  ret <4 x float> %2
}

define <8 x float> @stack_fold_shufps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_shufps_ymm
  ;CHECK:       vshufps $-108, {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <8 x float> %a0, <8 x float> %a1, <8 x i32> <i32 0, i32 1, i32 9, i32 10, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x float> %2
}

define <2 x double> @stack_fold_sqrtpd(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_sqrtpd
  ;CHECK:       vsqrtpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.sqrt.pd(<2 x double> %a0)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.sqrt.pd(<2 x double>) nounwind readnone

define <4 x double> @stack_fold_sqrtpd_ymm(<4 x double> %a0) {
  ;CHECK-LABEL: stack_fold_sqrtpd_ymm
  ;CHECK:       vsqrtpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x double> @llvm.x86.avx.sqrt.pd.256(<4 x double> %a0)
  ret <4 x double> %2
}
declare <4 x double> @llvm.x86.avx.sqrt.pd.256(<4 x double>) nounwind readnone

define <4 x float> @stack_fold_sqrtps(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_sqrtps
  ;CHECK:       vsqrtps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.sqrt.ps(<4 x float> %a0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.sqrt.ps(<4 x float>) nounwind readnone

define <8 x float> @stack_fold_sqrtps_ymm(<8 x float> %a0) {
  ;CHECK-LABEL: stack_fold_sqrtps_ymm
  ;CHECK:       vsqrtps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x float> @llvm.x86.avx.sqrt.ps.256(<8 x float> %a0)
  ret <8 x float> %2
}
declare <8 x float> @llvm.x86.avx.sqrt.ps.256(<8 x float>) nounwind readnone

define double @stack_fold_sqrtsd(double %a0) {
  ;CHECK-LABEL: stack_fold_sqrtsd
  ;CHECK:       vsqrtsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call double @llvm.sqrt.f64(double %a0)
  ret double %2
}
declare double @llvm.sqrt.f64(double) nounwind readnone

define <2 x double> @stack_fold_sqrtsd_int(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_sqrtsd_int
  ;CHECK:       vsqrtsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x double> @llvm.x86.sse2.sqrt.sd(<2 x double> %a0)
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse2.sqrt.sd(<2 x double>) nounwind readnone

define float @stack_fold_sqrtss(float %a0) {
  ;CHECK-LABEL: stack_fold_sqrtss
  ;CHECK:       vsqrtss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call float @llvm.sqrt.f32(float %a0)
  ret float %2
}
declare float @llvm.sqrt.f32(float) nounwind readnone

define <4 x float> @stack_fold_sqrtss_int(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_sqrtss_int
  ;CHECK:       vsqrtss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x float> @llvm.x86.sse.sqrt.ss(<4 x float> %a0)
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.sqrt.ss(<4 x float>) nounwind readnone

define <2 x double> @stack_fold_subpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_subpd
  ;CHECK:       vsubpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fsub <2 x double> %a0, %a1
  ret <2 x double> %2
}

define <4 x double> @stack_fold_subpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_subpd_ymm
  ;CHECK:       vsubpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fsub <4 x double> %a0, %a1
  ret <4 x double> %2
}

define <4 x float> @stack_fold_subps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_subps
  ;CHECK:       vsubps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fsub <4 x float> %a0, %a1
  ret <4 x float> %2
}

define <8 x float> @stack_fold_subps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_subps_ymm
  ;CHECK:       vsubps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fsub <8 x float> %a0, %a1
  ret <8 x float> %2
}

define double @stack_fold_subsd(double %a0, double %a1) {
  ;CHECK-LABEL: stack_fold_subsd
  ;CHECK:       vsubsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fsub double %a0, %a1
  ret double %2
}

; TODO stack_fold_subsd_int
declare <2 x double> @llvm.x86.sse2.sub.sd(<2 x double>, <2 x double>) nounwind readnone

define float @stack_fold_subss(float %a0, float %a1) {
  ;CHECK-LABEL: stack_fold_subss
  ;CHECK:       vsubss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fsub float %a0, %a1
  ret float %2
}

; TODO stack_fold_subss_int
declare <4 x float> @llvm.x86.sse.sub.ss(<4 x float>, <4 x float>) nounwind readnone

; TODO stack_fold_testpd
declare i32 @llvm.x86.avx.vtestc.pd(<2 x double>, <2 x double>) nounwind readnone

; TODO stack_fold_testpd_ymm
declare i32 @llvm.x86.avx.vtestc.pd.256(<4 x double>, <4 x double>) nounwind readnone

; TODO stack_fold_testps
declare i32 @llvm.x86.avx.vtestc.ps(<4 x float>, <4 x float>) nounwind readnone

; TODO stack_fold_testps_ymm
declare i32 @llvm.x86.avx.vtestc.ps.256(<8 x float>, <8 x float>) nounwind readnone

define i32 @stack_fold_ucomisd(double %a0, double %a1) {
  ;CHECK-LABEL: stack_fold_ucomisd
  ;CHECK:       vucomisd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp ueq double %a0, %a1
  %3 = select i1 %2, i32 1, i32 -1
  ret i32 %3
}

define i32 @stack_fold_ucomisd_int(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_ucomisd_int
  ;CHECK:       vucomisd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i32 @llvm.x86.sse2.ucomieq.sd(<2 x double> %a0, <2 x double> %a1)
  ret i32 %2
}
declare i32 @llvm.x86.sse2.ucomieq.sd(<2 x double>, <2 x double>) nounwind readnone

define i32 @stack_fold_ucomiss(float %a0, float %a1) {
  ;CHECK-LABEL: stack_fold_ucomiss
  ;CHECK:       vucomiss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 4-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = fcmp ueq float %a0, %a1
  %3 = select i1 %2, i32 1, i32 -1
  ret i32 %3
}

define i32 @stack_fold_ucomiss_int(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_ucomiss_int
  ;CHECK:       vucomiss {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call i32 @llvm.x86.sse.ucomieq.ss(<4 x float> %a0, <4 x float> %a1)
  ret i32 %2
}
declare i32 @llvm.x86.sse.ucomieq.ss(<4 x float>, <4 x float>) nounwind readnone

define <2 x double> @stack_fold_unpckhpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_unpckhpd
  ;CHECK:       vunpckhpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <2 x double> %a0, <2 x double> %a1, <2 x i32> <i32 1, i32 3>
  ret <2 x double> %2
}

define <4 x double> @stack_fold_unpckhpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_unpckhpd_ymm
  ;CHECK:       vunpckhpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x double> %a0, <4 x double> %a1, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  ret <4 x double> %2
}

define <4 x float> @stack_fold_unpckhps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_unpckhps
  ;CHECK:       vunpckhps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> %a1, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  ret <4 x float> %2
}

define <8 x float> @stack_fold_unpckhps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_unpckhps_ymm
  ;CHECK:       vunpckhps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <8 x float> %a0, <8 x float> %a1, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  ret <8 x float> %2
}

define <2 x double> @stack_fold_unpcklpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_unpcklpd
  ;CHECK:       vunpcklpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <2 x double> %a0, <2 x double> %a1, <2 x i32> <i32 0, i32 2>
  ret <2 x double> %2
}

define <4 x double> @stack_fold_unpcklpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_unpcklpd_ymm
  ;CHECK:       vunpcklpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x double> %a0, <4 x double> %a1, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  ret <4 x double> %2
}

define <4 x float> @stack_fold_unpcklps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_unpcklps
  ;CHECK:       vunpcklps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x float> %a0, <4 x float> %a1, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  ret <4 x float> %2
}

define <8 x float> @stack_fold_unpcklps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_unpcklps_ymm
  ;CHECK:       vunpcklps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <8 x float> %a0, <8 x float> %a1, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  ret <8 x float> %2
}

define <2 x double> @stack_fold_xorpd(<2 x double> %a0, <2 x double> %a1) {
  ;CHECK-LABEL: stack_fold_xorpd
  ;CHECK:       vxorpd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <2 x double> %a0 to <2 x i64>
  %3 = bitcast <2 x double> %a1 to <2 x i64>
  %4 = xor <2 x i64> %2, %3
  %5 = bitcast <2 x i64> %4 to <2 x double>
  ; fadd forces execution domain
  %6 = fadd <2 x double> %5, <double 0x0, double 0x0>
  ret <2 x double> %6
}

define <4 x double> @stack_fold_xorpd_ymm(<4 x double> %a0, <4 x double> %a1) {
  ;CHECK-LABEL: stack_fold_xorpd_ymm
  ;CHECK:       vxorpd {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <4 x double> %a0 to <4 x i64>
  %3 = bitcast <4 x double> %a1 to <4 x i64>
  %4 = xor <4 x i64> %2, %3
  %5 = bitcast <4 x i64> %4 to <4 x double>
  ; fadd forces execution domain
  %6 = fadd <4 x double> %5, <double 0x0, double 0x0, double 0x0, double 0x0>
  ret <4 x double> %6
}

define <4 x float> @stack_fold_xorps(<4 x float> %a0, <4 x float> %a1) {
  ;CHECK-LABEL: stack_fold_xorps
  ;CHECK:       vxorps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <4 x float> %a0 to <2 x i64>
  %3 = bitcast <4 x float> %a1 to <2 x i64>
  %4 = xor <2 x i64> %2, %3
  %5 = bitcast <2 x i64> %4 to <4 x float>
  ret <4 x float> %5
}

define <8 x float> @stack_fold_xorps_ymm(<8 x float> %a0, <8 x float> %a1) {
  ;CHECK-LABEL: stack_fold_xorps_ymm
  ;CHECK:       vxorps {{-?[0-9]*}}(%rsp), {{%ymm[0-9][0-9]*}}, {{%ymm[0-9][0-9]*}} {{.*#+}} 32-byte Folded Reload
  %1 = tail call <4 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = bitcast <8 x float> %a0 to <4 x i64>
  %3 = bitcast <8 x float> %a1 to <4 x i64>
  %4 = xor <4 x i64> %2, %3
  %5 = bitcast <4 x i64> %4 to <8 x float>
  ret <8 x float> %5
}
