; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+3dnow | FileCheck %s

define x86_mmx @stack_fold_pavgusb(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pavgusb
  ;CHECK:       pavgusb {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.3dnow.pavgusb(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.3dnow.pavgusb(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pf2id(x86_mmx %a) {
  ;CHECK-LABEL: stack_fold_pf2id
  ;CHECK:       pf2id {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm1},~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.3dnow.pf2id(x86_mmx %a) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.3dnow.pf2id(x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pf2iw(x86_mmx %a) {
  ;CHECK-LABEL: stack_fold_pf2iw
  ;CHECK:       pf2iw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm1},~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.3dnowa.pf2iw(x86_mmx %a) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.3dnowa.pf2iw(x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pfacc(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pfacc
  ;CHECK:       pfacc {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.3dnow.pfacc(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.3dnow.pfacc(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pfadd(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pfadd
  ;CHECK:       pfadd {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.3dnow.pfadd(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.3dnow.pfadd(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pfcmpeq(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pfcmpeq
  ;CHECK:       pfcmpeq {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.3dnow.pfcmpeq(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.3dnow.pfcmpeq(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pfcmpge(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pfcmpge
  ;CHECK:       pfcmpge {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.3dnow.pfcmpge(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.3dnow.pfcmpge(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pfcmpgt(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pfcmpgt
  ;CHECK:       pfcmpgt {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.3dnow.pfcmpgt(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.3dnow.pfcmpgt(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pfmax(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pfmax
  ;CHECK:       pfmax {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.3dnow.pfmax(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.3dnow.pfmax(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pfmin(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pfmin
  ;CHECK:       pfmin {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.3dnow.pfmin(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.3dnow.pfmin(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pfmul(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pfmul
  ;CHECK:       pfmul {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.3dnow.pfmul(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.3dnow.pfmul(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pfnacc(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pfnacc
  ;CHECK:       pfnacc {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.3dnowa.pfnacc(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.3dnowa.pfnacc(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pfpnacc(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pfpnacc
  ;CHECK:       pfpnacc {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.3dnowa.pfpnacc(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.3dnowa.pfpnacc(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pfrcp(x86_mmx %a) {
  ;CHECK-LABEL: stack_fold_pfrcp
  ;CHECK:       pfrcp {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm1},~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.3dnow.pfrcp(x86_mmx %a) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.3dnow.pfrcp(x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pfrcpit1(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pfrcpit1
  ;CHECK:       pfrcpit1 {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.3dnow.pfrcpit1(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.3dnow.pfrcpit1(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pfrcpit2(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pfrcpit2
  ;CHECK:       pfrcpit2 {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.3dnow.pfrcpit2(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.3dnow.pfrcpit2(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pfrsqit1(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pfrsqit1
  ;CHECK:       pfrsqit1 {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.3dnow.pfrsqit1(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.3dnow.pfrsqit1(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pfrsqrt(x86_mmx %a) {
  ;CHECK-LABEL: stack_fold_pfrsqrt
  ;CHECK:       pfrsqrt {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm1},~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.3dnow.pfrsqrt(x86_mmx %a) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.3dnow.pfrsqrt(x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pfsub(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pfsub
  ;CHECK:       pfsub {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.3dnow.pfsub(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.3dnow.pfsub(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pfsubr(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pfsubr
  ;CHECK:       pfsubr {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.3dnow.pfsubr(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.3dnow.pfsubr(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pi2fd(x86_mmx %a) {
  ;CHECK-LABEL: stack_fold_pi2fd
  ;CHECK:       pi2fd {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm1},~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.3dnow.pi2fd(x86_mmx %a) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.3dnow.pi2fd(x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pi2fw(x86_mmx %a) {
  ;CHECK-LABEL: stack_fold_pi2fw
  ;CHECK:       pi2fw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm1},~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.3dnowa.pi2fw(x86_mmx %a) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.3dnowa.pi2fw(x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pmulhrw(x86_mmx %a, x86_mmx %b) {
  ;CHECK-LABEL: stack_fold_pmulhrw
  ;CHECK:       pmulhrw {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.3dnow.pmulhrw(x86_mmx %a, x86_mmx %b) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.3dnow.pmulhrw(x86_mmx, x86_mmx) nounwind readnone

define x86_mmx @stack_fold_pswapd(x86_mmx %a) {
  ;CHECK-LABEL: stack_fold_pswapd
  ;CHECK:       pswapd {{-?[0-9]*}}(%rsp), {{%mm[0-7]}} {{.*#+}} 8-byte Folded Reload
  %1 = tail call x86_mmx asm sideeffect "nop", "=y,~{mm1},~{mm2},~{mm3},~{mm4},~{mm5},~{mm6},~{mm7}"()
  %2 = call x86_mmx @llvm.x86.3dnowa.pswapd(x86_mmx %a) nounwind readnone
  ret x86_mmx %2
}
declare x86_mmx @llvm.x86.3dnowa.pswapd(x86_mmx) nounwind readnone
