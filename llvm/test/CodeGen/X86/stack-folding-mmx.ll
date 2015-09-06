; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+mmx,+ssse3 | FileCheck %s

define x86_mmx @stack_fold_cvtpd2pi(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvtpd2pi
  ;CHECK:       cvtpd2pi {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call x86_mmx @llvm.x86.sse.cvtpd2pi(<2 x double> %a0) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.sse.cvtpd2pi(<2 x double>) nounwind readnone

define <2 x double> @stack_fold_cvtpi2pd(x86_mmx %a0) {
  ;CHECK-LABEL: stack_fold_cvtpi2pd
  ;CHECK:       cvtpi2pd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm1},~{mm1},~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call <2 x double> @llvm.x86.sse.cvtpi2pd(x86_mmx %a0) nounwind readnone
  ret <2 x double> %2
}
declare <2 x double> @llvm.x86.sse.cvtpi2pd(x86_mmx) nounwind readnone

define <4 x float> @stack_fold_cvtpi2ps(<4 x float> %a0, x86_mmx %a1) {
  ;CHECK-LABEL: stack_fold_cvtpi2ps
  ;CHECK:       cvtpi2ps {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm1},~{mm1},~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call <4 x float> @llvm.x86.sse.cvtpi2ps(<4 x float> %a0, x86_mmx %a1) nounwind readnone
  ret <4 x float> %2
}
declare <4 x float> @llvm.x86.sse.cvtpi2ps(<4 x float>, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_cvtps2pi(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvtps2pi
  ;CHECK:       cvtps2pi {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call x86_mmx @llvm.x86.sse.cvtps2pi(<4 x float> %a0) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.sse.cvtps2pi(<4 x float>) nounwind readnone

define x86_mmx @stack_fold_cvttpd2pi(<2 x double> %a0) {
  ;CHECK-LABEL: stack_fold_cvttpd2pi
  ;CHECK:       cvttpd2pi {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call x86_mmx @llvm.x86.sse.cvttpd2pi(<2 x double> %a0) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.sse.cvttpd2pi(<2 x double>) nounwind readnone

define x86_mmx @stack_fold_cvttps2pi(<4 x float> %a0) {
  ;CHECK-LABEL: stack_fold_cvttps2pi
  ;CHECK:       cvttps2pi {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call x86_mmx @llvm.x86.sse.cvttps2pi(<4 x float> %a0) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.sse.cvttps2pi(<4 x float>) nounwind readnone

; TODO stack_fold_movd_load
; TODO stack_fold_movd_store
; TODO stack_fold_movq_load
; TODO stack_fold_movq_store

define x86_mmx @stack_fold_pabsb(x86_mmx %a0) {
  ;CHECK-LABEL: stack_fold_pabsb
  ;CHECK:       pabsb {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm1},~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.ssse3.pabs.b(x86_mmx %a0) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.ssse3.pabs.b(x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pabsd(x86_mmx %a0) {
  ;CHECK-LABEL: stack_fold_pabsd
  ;CHECK:       pabsd {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm1},~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.ssse3.pabs.d(x86_mmx %a0) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.ssse3.pabs.d(x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pabsw(x86_mmx %a0) {
  ;CHECK-LABEL: stack_fold_pabsw
  ;CHECK:       pabsw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm1},~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.ssse3.pabs.w(x86_mmx %a0) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.ssse3.pabs.w(x86_mmx) nounwind readnone

define x86_mmx @stack_fold_packssdw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_packssdw
  ;CHECK:       packssdw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.packssdw(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.packssdw(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_packsswb(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_packsswb
  ;CHECK:       packsswb {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.packsswb(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.packsswb(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_packuswb(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_packuswb
  ;CHECK:       packuswb {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.packuswb(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.packuswb(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_paddb(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_paddb
  ;CHECK:       paddb {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.padd.b(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.padd.b(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_paddd(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_paddd
  ;CHECK:       paddd {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.padd.d(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.padd.d(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_paddq(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_paddq
  ;CHECK:       paddq {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.padd.q(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.padd.q(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_paddsb(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_paddsb
  ;CHECK:       paddsb {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.padds.b(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.padds.b(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_paddsw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_paddsw
  ;CHECK:       paddsw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.padds.w(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.padds.w(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_paddusb(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_paddusb
  ;CHECK:       paddusb {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.paddus.b(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.paddus.b(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_paddusw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_paddusw
  ;CHECK:       paddusw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.paddus.w(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.paddus.w(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_paddw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_paddw
  ;CHECK:       paddw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.padd.w(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.padd.w(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_palignr(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_palignr
  ;CHECK:       palignr $1, {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.palignr.b(x86_mmx %a, x86_mmx %b, i8 1) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.palignr.b(x86_mmx, x86_mmx, i8) nounwind readnone

define x86_mmx @stack_fold_pand(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pand
  ;CHECK:       pand {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.pand(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.pand(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pandn(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pandn
  ;CHECK:       pandn {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.pandn(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.pandn(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pavgb(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pavgb
  ;CHECK:       pavgb {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.pavg.b(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.pavg.b(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pavgw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pavgw
  ;CHECK:       pavgw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.pavg.w(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.pavg.w(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pcmpeqb(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pcmpeqb
  ;CHECK:       pcmpeqb {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.pcmpeq.b(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.pcmpeq.b(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pcmpeqd(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pcmpeqd
  ;CHECK:       pcmpeqd {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.pcmpeq.d(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.pcmpeq.d(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pcmpeqw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pcmpeqw
  ;CHECK:       pcmpeqw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.pcmpeq.w(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.pcmpeq.w(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pcmpgtb(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pcmpgtb
  ;CHECK:       pcmpgtb {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.pcmpgt.b(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.pcmpgt.b(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pcmpgtd(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pcmpgtd
  ;CHECK:       pcmpgtd {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.pcmpgt.d(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.pcmpgt.d(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pcmpgtw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pcmpgtw
  ;CHECK:       pcmpgtw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.pcmpgt.w(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.pcmpgt.w(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_phaddd(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_phaddd
  ;CHECK:       phaddd {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.ssse3.phadd.d(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.ssse3.phadd.d(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_phaddsw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_phaddsw
  ;CHECK:       phaddsw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.ssse3.phadd.sw(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.ssse3.phadd.sw(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_phaddw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_phaddw
  ;CHECK:       phaddw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.ssse3.phadd.w(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.ssse3.phadd.w(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_phsubd(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_phsubd
  ;CHECK:       phsubd {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.ssse3.phsub.d(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.ssse3.phsub.d(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_phsubsw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_phsubsw
  ;CHECK:       phsubsw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.ssse3.phsub.sw(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.ssse3.phsub.sw(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_phsubw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_phsubw
  ;CHECK:       phsubw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.ssse3.phsub.w(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.ssse3.phsub.w(x86_mmx, x86_mmx) nounwind readnone

; TODO stack_fold_pinsrw

define x86_mmx @stack_fold_pmaddubsw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pmaddubsw
  ;CHECK:       pmaddubsw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.ssse3.pmadd.ub.sw(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.ssse3.pmadd.ub.sw(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pmaddwd(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pmaddwd
  ;CHECK:       pmaddwd {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.pmadd.wd(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.pmadd.wd(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pmaxsw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pmaxsw
  ;CHECK:       pmaxsw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.pmaxs.w(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.pmaxs.w(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pmaxub(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pmaxub
  ;CHECK:       pmaxub {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.pmaxu.b(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.pmaxu.b(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pminsw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pminsw
  ;CHECK:       pminsw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.pmins.w(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.pmins.w(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pminub(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pminub
  ;CHECK:       pminub {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.pminu.b(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.pminu.b(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pmulhrsw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pmulhrsw
  ;CHECK:       pmulhrsw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.ssse3.pmul.hr.sw(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.ssse3.pmul.hr.sw(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pmulhuw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pmulhuw
  ;CHECK:       pmulhuw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.pmulhu.w(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.pmulhu.w(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pmulhw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pmulhw
  ;CHECK:       pmulhw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.pmulh.w(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.pmulh.w(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pmullw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pmullw
  ;CHECK:       pmullw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.pmull.w(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.pmull.w(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pmuludq(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pmuludq
  ;CHECK:       pmuludq {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.pmulu.dq(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.pmulu.dq(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_por(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_por
  ;CHECK:       por {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.por(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.por(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_psadbw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_psadbw
  ;CHECK:       psadbw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.psad.bw(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.psad.bw(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pshufb(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pshufb
  ;CHECK:       pshufb {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm1},~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.ssse3.pshuf.b(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.ssse3.pshuf.b(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pshufw(x86_mmx %a) {
  ;CHECK-LABEL: stack_fold_pshufw
  ;CHECK:       pshufw $1, {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm1},~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.sse.pshuf.w(x86_mmx %a, i8 1) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.sse.pshuf.w(x86_mmx, i8) nounwind readnone

define x86_mmx @stack_fold_psignb(x86_mmx %a0, x86_mmx %a1) {
  ;CHECK-LABEL: stack_fold_psignb
  ;CHECK:       psignb {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.ssse3.psign.b(x86_mmx %a0, x86_mmx %a1) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.ssse3.psign.b(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_psignd(x86_mmx %a0, x86_mmx %a1) {
  ;CHECK-LABEL: stack_fold_psignd
  ;CHECK:       psignd {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.ssse3.psign.d(x86_mmx %a0, x86_mmx %a1) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.ssse3.psign.d(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_psignw(x86_mmx %a0, x86_mmx %a1) {
  ;CHECK-LABEL: stack_fold_psignw
  ;CHECK:       psignw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.ssse3.psign.w(x86_mmx %a0, x86_mmx %a1) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.ssse3.psign.w(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pslld(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pslld
  ;CHECK:       pslld {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.psll.d(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.psll.d(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_psllq(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_psllq
  ;CHECK:       psllq {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.psll.q(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.psll.q(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_psllw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_psllw
  ;CHECK:       psllw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.psll.w(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.psll.w(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_psrad(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_psrad
  ;CHECK:       psrad {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.psra.d(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.psra.d(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_psraw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_psraw
  ;CHECK:       psraw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.psra.w(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.psra.w(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_psrld(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_psrld
  ;CHECK:       psrld {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.psrl.d(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.psrl.d(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_psrlq(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_psrlq
  ;CHECK:       psrlq {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.psrl.q(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.psrl.q(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_psrlw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_psrlw
  ;CHECK:       psrlw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.psrl.w(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.psrl.w(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_psubb(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_psubb
  ;CHECK:       psubb {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.psub.b(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.psub.b(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_psubd(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_psubd
  ;CHECK:       psubd {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.psub.d(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.psub.d(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_psubq(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_psubq
  ;CHECK:       psubq {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.psub.q(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.psub.q(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_psubsb(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_psubsb
  ;CHECK:       psubsb {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.psubs.b(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.psubs.b(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_psubsw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_psubsw
  ;CHECK:       psubsw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.psubs.w(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.psubs.w(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_psubusb(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_psubusb
  ;CHECK:       psubusb {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.psubus.b(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.psubus.b(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_psubusw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_psubusw
  ;CHECK:       psubusw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.psubus.w(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.psubus.w(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_psubw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_psubw
  ;CHECK:       psubw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.psub.w(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.psub.w(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_punpckhbw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_punpckhbw
  ;CHECK:       punpckhbw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.punpckhbw(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.punpckhbw(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_punpckhdq(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_punpckhdq
  ;CHECK:       punpckhdq {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.punpckhdq(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.punpckhdq(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_punpckhwd(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_punpckhwd
  ;CHECK:       punpckhwd {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.punpckhwd(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.punpckhwd(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_punpcklbw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_punpcklbw
  ;CHECK:       punpcklbw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.punpcklbw(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.punpcklbw(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_punpckldq(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_punpckldq
  ;CHECK:       punpckldq {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.punpckldq(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.punpckldq(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_punpcklwd(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_punpcklwd
  ;CHECK:       punpcklwd {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.punpcklwd(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.punpcklwd(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pxor(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pxor
  ;CHECK:       pxor {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.mmx.pxor(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.mmx.pxor(x86_mmx, x86_mmx) nounwind readnone
