; RUN: llc -O3 -disable-peephole -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+avx,+aes < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Stack reload folding tests.
;
; By including a nop call with sideeffects we can force a partial register spill of the
; relevant registers and check that the reload is correctly folded into the instruction.

define <2 x i64> @stack_fold_aesdec(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_aesdec
  ;CHECK:       vaesdec {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.aesni.aesdec(<2 x i64> %a0, <2 x i64> %a1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.aesni.aesdec(<2 x i64>, <2 x i64>) nounwind readnone

define <2 x i64> @stack_fold_aesdeclast(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_aesdeclast
  ;CHECK:       vaesdeclast {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.aesni.aesdeclast(<2 x i64> %a0, <2 x i64> %a1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.aesni.aesdeclast(<2 x i64>, <2 x i64>) nounwind readnone

define <2 x i64> @stack_fold_aesenc(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_aesenc
  ;CHECK:       vaesenc {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.aesni.aesenc(<2 x i64> %a0, <2 x i64> %a1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.aesni.aesenc(<2 x i64>, <2 x i64>) nounwind readnone

define <2 x i64> @stack_fold_aesenclast(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_aesenclast
  ;CHECK:       vaesenclast {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.aesni.aesenclast(<2 x i64> %a0, <2 x i64> %a1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.aesni.aesenclast(<2 x i64>, <2 x i64>) nounwind readnone

define <2 x i64> @stack_fold_aesimc(<2 x i64> %a0) {
  ;CHECK-LABEL: stack_fold_aesimc
  ;CHECK:       vaesimc {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.aesni.aesimc(<2 x i64> %a0)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.aesni.aesimc(<2 x i64>) nounwind readnone

define <2 x i64> @stack_fold_aeskeygenassist(<2 x i64> %a0) {
  ;CHECK-LABEL: stack_fold_aeskeygenassist
  ;CHECK:       vaeskeygenassist $7, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.aesni.aeskeygenassist(<2 x i64> %a0, i8 7)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.aesni.aeskeygenassist(<2 x i64>, i8) nounwind readnone

define <16 x i8> @stack_fold_pabsb(<16 x i8> %a0) {
  ;CHECK-LABEL: stack_fold_pabsb
  ;CHECK:       vpabsb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.ssse3.pabs.b.128(<16 x i8> %a0)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.ssse3.pabs.b.128(<16 x i8>) nounwind readnone

define <4 x i32> @stack_fold_pabsd(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_pabsd
  ;CHECK:       vpabsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.ssse3.pabs.d.128(<4 x i32> %a0)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.ssse3.pabs.d.128(<4 x i32>) nounwind readnone

define <8 x i16> @stack_fold_pabsw(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_pabsw
  ;CHECK:       vpabsw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.ssse3.pabs.w.128(<8 x i16> %a0)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.ssse3.pabs.w.128(<8 x i16>) nounwind readnone

define <8 x i16> @stack_fold_packssdw(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_packssdw
  ;CHECK:       vpackssdw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse2.packssdw.128(<4 x i32> %a0, <4 x i32> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse2.packssdw.128(<4 x i32>, <4 x i32>) nounwind readnone

define <16 x i8> @stack_fold_packsswb(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_packsswb
  ;CHECK:       vpacksswb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.sse2.packsswb.128(<8 x i16> %a0, <8 x i16> %a1)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.sse2.packsswb.128(<8 x i16>, <8 x i16>) nounwind readnone

define <8 x i16> @stack_fold_packusdw(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_packusdw
  ;CHECK:       vpackusdw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse41.packusdw(<4 x i32> %a0, <4 x i32> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse41.packusdw(<4 x i32>, <4 x i32>) nounwind readnone

define <16 x i8> @stack_fold_packuswb(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_packuswb
  ;CHECK:       vpackuswb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.sse2.packuswb.128(<8 x i16> %a0, <8 x i16> %a1)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.sse2.packuswb.128(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i8> @stack_fold_paddb(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_paddb
  ;CHECK:       vpaddb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = add <16 x i8> %a0, %a1
  ret <16 x i8> %2
}

define <4 x i32> @stack_fold_paddd(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_paddd
  ;CHECK:       vpaddd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = add <4 x i32> %a0, %a1
  ret <4 x i32> %2
}

define <2 x i64> @stack_fold_paddq(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_paddq
  ;CHECK:       vpaddq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = add <2 x i64> %a0, %a1
  ret <2 x i64> %2
}

define <16 x i8> @stack_fold_paddsb(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_paddsb
  ;CHECK:       vpaddsb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.sse2.padds.b(<16 x i8> %a0, <16 x i8> %a1)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.sse2.padds.b(<16 x i8>, <16 x i8>) nounwind readnone

define <8 x i16> @stack_fold_paddsw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_paddsw
  ;CHECK:       vpaddsw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse2.padds.w(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse2.padds.w(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i8> @stack_fold_paddusb(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_paddusb
  ;CHECK:       vpaddusb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.sse2.paddus.b(<16 x i8> %a0, <16 x i8> %a1)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.sse2.paddus.b(<16 x i8>, <16 x i8>) nounwind readnone

define <8 x i16> @stack_fold_paddusw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_paddusw
  ;CHECK:       vpaddusw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse2.paddus.w(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse2.paddus.w(<8 x i16>, <8 x i16>) nounwind readnone

define <8 x i16> @stack_fold_paddw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_paddw
  ;CHECK:       vpaddw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = add <8 x i16> %a0, %a1
  ret <8 x i16> %2
}

define <16 x i8> @stack_fold_palignr(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_palignr
  ;CHECK:       vpalignr $1, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <16 x i8> %a1, <16 x i8> %a0, <16 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16>
  ret <16 x i8> %2
}

define <16 x i8> @stack_fold_pand(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pand
  ;CHECK:       vpand {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = and <16 x i8> %a0, %a1
  ; add forces execution domain
  %3 = add <16 x i8> %2, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ret <16 x i8> %3
}

define <16 x i8> @stack_fold_pandn(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pandn
  ;CHECK:       vpandn {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = xor <16 x i8> %a0, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  %3 = and <16 x i8> %2, %a1
  ; add forces execution domain
  %4 = add <16 x i8> %3, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ret <16 x i8> %4
}

define <16 x i8> @stack_fold_pavgb(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pavgb
  ;CHECK:       vpavgb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.sse2.pavg.b(<16 x i8> %a0, <16 x i8> %a1)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.sse2.pavg.b(<16 x i8>, <16 x i8>) nounwind readnone

define <8 x i16> @stack_fold_pavgw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pavgw
  ;CHECK:       vpavgw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse2.pavg.w(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse2.pavg.w(<8 x i16>, <8 x i16>) nounwind readnone

; TODO stack_fold_pblendvb
declare <16 x i8> @llvm.x86.sse41.pblendvb(<16 x i8>, <16 x i8>, <16 x i8>) nounwind readnone

define <8 x i16> @stack_fold_pblendw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pblendw
  ;CHECK:       vpblendw $7, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse41.pblendw(<8 x i16> %a0, <8 x i16> %a1, i8 7)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse41.pblendw(<8 x i16>, <8 x i16>, i8) nounwind readnone

; TODO stack_fold_pclmulqdq
declare <2 x i64> @llvm.x86.pclmulqdq(<2 x i64>, <2 x i64>, i8) nounwind readnone

define <16 x i8> @stack_fold_pcmpeqb(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pcmpeqb
  ;CHECK:       vpcmpeqb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = icmp eq <16 x i8> %a0, %a1
  %3 = sext <16 x i1> %2 to <16 x i8>
  ret <16 x i8> %3
}

define <4 x i32> @stack_fold_pcmpeqd(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pcmpeqd
  ;CHECK:       vpcmpeqd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = icmp eq <4 x i32> %a0, %a1
  %3 = sext <4 x i1> %2 to <4 x i32>
  ret <4 x i32> %3
}

define <2 x i64> @stack_fold_pcmpeqq(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_pcmpeqq
  ;CHECK:       vpcmpeqq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = icmp eq <2 x i64> %a0, %a1
  %3 = sext <2 x i1> %2 to <2 x i64>
  ret <2 x i64> %3
}

define <8 x i16> @stack_fold_pcmpeqw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pcmpeqw
  ;CHECK:       vpcmpeqw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = icmp eq <8 x i16> %a0, %a1
  %3 = sext <8 x i1> %2 to <8 x i16>
  ret <8 x i16> %3
}

; TODO stack_fold_pcmpestri
declare i32 @llvm.x86.sse42.pcmpestri128(<16 x i8>, i32, <16 x i8>, i32, i8) nounwind readnone

; TODO stack_fold_pcmpestrm
declare <16 x i8> @llvm.x86.sse42.pcmpestrm128(<16 x i8>, i32, <16 x i8>, i32, i8) nounwind readnone

define <16 x i8> @stack_fold_pcmpgtb(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pcmpgtb
  ;CHECK:       vpcmpgtb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = icmp sgt <16 x i8> %a0, %a1
  %3 = sext <16 x i1> %2 to <16 x i8>
  ret <16 x i8> %3
}

define <4 x i32> @stack_fold_pcmpgtd(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pcmpgtd
  ;CHECK:       vpcmpgtd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = icmp sgt <4 x i32> %a0, %a1
  %3 = sext <4 x i1> %2 to <4 x i32>
  ret <4 x i32> %3
}

define <2 x i64> @stack_fold_pcmpgtq(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_pcmpgtq
  ;CHECK:       vpcmpgtq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = icmp sgt <2 x i64> %a0, %a1
  %3 = sext <2 x i1> %2 to <2 x i64>
  ret <2 x i64> %3
}

define <8 x i16> @stack_fold_pcmpgtw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pcmpgtw
  ;CHECK:       vpcmpgtw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = icmp sgt <8 x i16> %a0, %a1
  %3 = sext <8 x i1> %2 to <8 x i16>
  ret <8 x i16> %3
}

; TODO stack_fold_pcmpistri
declare i32 @llvm.x86.sse42.pcmpistri128(<16 x i8>, <16 x i8>, i8) nounwind readnone

; TODO stack_fold_pcmpistrm
declare <16 x i8> @llvm.x86.sse42.pcmpistrm128(<16 x i8>, <16 x i8>, i8) nounwind readnone

; TODO stack_fold_pextrb
; TODO stack_fold_pextrd
; TODO stack_fold_pextrq
; TODO stack_fold_pextrw

define <4 x i32> @stack_fold_phaddd(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_phaddd
  ;CHECK:       vphaddd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.ssse3.phadd.d.128(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.ssse3.phadd.d.128(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i16> @stack_fold_phaddsw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_phaddsw
  ;CHECK:       vphaddsw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.ssse3.phadd.sw.128(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.ssse3.phadd.sw.128(<8 x i16>, <8 x i16>) nounwind readnone

define <8 x i16> @stack_fold_phaddw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_phaddw
  ;CHECK:       vphaddw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.ssse3.phadd.w.128(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.ssse3.phadd.w.128(<8 x i16>, <8 x i16>) nounwind readnone

; TODO stack_fold_phminposuw
declare <8 x i16> @llvm.x86.sse41.phminposuw(<8 x i16>) nounwind readnone

define <4 x i32> @stack_fold_phsubd(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_phsubd
  ;CHECK:       vphsubd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.ssse3.phsub.d.128(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.ssse3.phsub.d.128(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i16> @stack_fold_phsubsw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_phsubsw
  ;CHECK:       vphsubsw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.ssse3.phsub.sw.128(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.ssse3.phsub.sw.128(<8 x i16>, <8 x i16>) nounwind readnone

define <8 x i16> @stack_fold_phsubw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_phsubw
  ;CHECK:       vphsubw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.ssse3.phsub.w.128(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.ssse3.phsub.w.128(<8 x i16>, <8 x i16>) nounwind readnone

; TODO stack_fold_pinsrb
; TODO stack_fold_pinsrd
; TODO stack_fold_pinsrq
; TODO stack_fold_pinsrw

define <8 x i16> @stack_fold_pmaddubsw(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pmaddubsw
  ;CHECK:       vpmaddubsw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.ssse3.pmadd.ub.sw.128(<16 x i8> %a0, <16 x i8> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.ssse3.pmadd.ub.sw.128(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @stack_fold_pmaddwd(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pmaddwd
  ;CHECK:       vpmaddwd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse2.pmadd.wd(<8 x i16> %a0, <8 x i16> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sse2.pmadd.wd(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i8> @stack_fold_pmaxsb(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pmaxsb
  ;CHECK:       vpmaxsb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.sse41.pmaxsb(<16 x i8> %a0, <16 x i8> %a1)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.sse41.pmaxsb(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @stack_fold_pmaxsd(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pmaxsd
  ;CHECK:       vpmaxsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse41.pmaxsd(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sse41.pmaxsd(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i16> @stack_fold_pmaxsw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pmaxsw
  ;CHECK:       vpmaxsw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse2.pmaxs.w(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse2.pmaxs.w(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i8> @stack_fold_pmaxub(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pmaxub
  ;CHECK:       vpmaxub {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.sse2.pmaxu.b(<16 x i8> %a0, <16 x i8> %a1)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.sse2.pmaxu.b(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @stack_fold_pmaxud(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pmaxud
  ;CHECK:       vpmaxud {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse41.pmaxud(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sse41.pmaxud(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i16> @stack_fold_pmaxuw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pmaxuw
  ;CHECK:       vpmaxuw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse41.pmaxuw(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse41.pmaxuw(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i8> @stack_fold_pminsb(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pminsb
  ;CHECK:       vpminsb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.sse41.pminsb(<16 x i8> %a0, <16 x i8> %a1)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.sse41.pminsb(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @stack_fold_pminsd(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pminsd
  ;CHECK:       vpminsd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse41.pminsd(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sse41.pminsd(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i16> @stack_fold_pminsw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pminsw
  ;CHECK:       vpminsw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse2.pmins.w(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse2.pmins.w(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i8> @stack_fold_pminub(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pminub
  ;CHECK:       vpminub {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.sse2.pminu.b(<16 x i8> %a0, <16 x i8> %a1)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.sse2.pminu.b(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @stack_fold_pminud(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pminud
  ;CHECK:       vpminud {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse41.pminud(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sse41.pminud(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i16> @stack_fold_pminuw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pminuw
  ;CHECK:       vpminuw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse41.pminuw(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse41.pminuw(<8 x i16>, <8 x i16>) nounwind readnone

; TODO stack_fold_pmovsxbd
declare <4 x i32> @llvm.x86.sse41.pmovsxbd(<16 x i8>) nounwind readnone

; TODO stack_fold_pmovsxbq
declare <2 x i64> @llvm.x86.sse41.pmovsxbq(<16 x i8>) nounwind readnone

; TODO stack_fold_pmovsxbw
declare <8 x i16> @llvm.x86.sse41.pmovsxbw(<16 x i8>) nounwind readnone

; TODO stack_fold_pmovsxdq
declare <2 x i64> @llvm.x86.sse41.pmovsxdq(<4 x i32>) nounwind readnone

; TODO stack_fold_pmovsxwd
declare <4 x i32> @llvm.x86.sse41.pmovsxwd(<8 x i16>) nounwind readnone

; TODO stack_fold_pmovsxwq
declare <2 x i64> @llvm.x86.sse41.pmovsxwq(<8 x i16>) nounwind readnone

; TODO stack_fold_pmovzxbd
declare <4 x i32> @llvm.x86.sse41.pmovzxbd(<16 x i8>) nounwind readnone

; TODO stack_fold_pmovzxbq
declare <2 x i64> @llvm.x86.sse41.pmovzxbq(<16 x i8>) nounwind readnone

; TODO stack_fold_pmovzxbw
declare <8 x i16> @llvm.x86.sse41.pmovzxbw(<16 x i8>) nounwind readnone

; TODO stack_fold_pmovzxdq
declare <2 x i64> @llvm.x86.sse41.pmovzxdq(<4 x i32>) nounwind readnone

; TODO stack_fold_pmovzxwd
declare <4 x i32> @llvm.x86.sse41.pmovzxwd(<8 x i16>) nounwind readnone

; TODO stack_fold_pmovzxwq
declare <2 x i64> @llvm.x86.sse41.pmovzxwq(<8 x i16>) nounwind readnone

define <2 x i64> @stack_fold_pmuldq(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pmuldq
  ;CHECK:       vpmuldq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.sse41.pmuldq(<4 x i32> %a0, <4 x i32> %a1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.sse41.pmuldq(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i16> @stack_fold_pmulhrsw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pmulhrsw
  ;CHECK:       vpmulhrsw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.ssse3.pmul.hr.sw.128(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.ssse3.pmul.hr.sw.128(<8 x i16>, <8 x i16>) nounwind readnone

define <8 x i16> @stack_fold_pmulhuw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pmulhuw
  ;CHECK:       vpmulhuw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse2.pmulhu.w(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse2.pmulhu.w(<8 x i16>, <8 x i16>) nounwind readnone

define <8 x i16> @stack_fold_pmulhw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pmulhw
  ;CHECK:       vpmulhw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse2.pmulh.w(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse2.pmulh.w(<8 x i16>, <8 x i16>) nounwind readnone

define <4 x i32> @stack_fold_pmulld(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pmulld
  ;CHECK:       vpmulld {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = mul <4 x i32> %a0, %a1
  ret <4 x i32> %2
}

define <8 x i16> @stack_fold_pmullw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_pmullw
  ;CHECK:       vpmullw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = mul <8 x i16> %a0, %a1
  ret <8 x i16> %2
}

define <2 x i64> @stack_fold_pmuludq(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pmuludq
  ;CHECK:       vpmuludq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.sse2.pmulu.dq(<4 x i32> %a0, <4 x i32> %a1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.sse2.pmulu.dq(<4 x i32>, <4 x i32>) nounwind readnone

define <16 x i8> @stack_fold_por(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_por
  ;CHECK:       vpor {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = or <16 x i8> %a0, %a1
  ; add forces execution domain
  %3 = add <16 x i8> %2, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ret <16 x i8> %3
}

define <2 x i64> @stack_fold_psadbw(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_psadbw
  ;CHECK:       vpsadbw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.sse2.psad.bw(<16 x i8> %a0, <16 x i8> %a1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.sse2.psad.bw(<16 x i8>, <16 x i8>) nounwind readnone

define <16 x i8> @stack_fold_pshufb(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pshufb
  ;CHECK:       vpshufb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8> %a0, <16 x i8> %a1)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @stack_fold_pshufd(<4 x i32> %a0) {
  ;CHECK-LABEL: stack_fold_pshufd
  ;CHECK:       vpshufd $27, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x i32> %a0, <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x i32> %2
}

define <8 x i16> @stack_fold_pshufhw(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_pshufhw
  ;CHECK:       vpshufhw $11, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <8 x i16> %a0, <8 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 7, i32 6, i32 4, i32 4>
  ret <8 x i16> %2
}

define <8 x i16> @stack_fold_pshuflw(<8 x i16> %a0) {
  ;CHECK-LABEL: stack_fold_pshuflw
  ;CHECK:       vpshuflw $27, {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <8 x i16> %a0, <8 x i16> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %2
}

define <16 x i8> @stack_fold_psignb(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_psignb
  ;CHECK:       vpsignb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.ssse3.psign.b.128(<16 x i8> %a0, <16 x i8> %a1)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.ssse3.psign.b.128(<16 x i8>, <16 x i8>) nounwind readnone

define <4 x i32> @stack_fold_psignd(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_psignd
  ;CHECK:       vpsignd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.ssse3.psign.d.128(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.ssse3.psign.d.128(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i16> @stack_fold_psignw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psignw
  ;CHECK:       vpsignw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.ssse3.psign.w.128(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.ssse3.psign.w.128(<8 x i16>, <8 x i16>) nounwind readnone

define <4 x i32> @stack_fold_pslld(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_pslld
  ;CHECK:       vpslld {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse2.psll.d(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sse2.psll.d(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @stack_fold_psllq(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_psllq
  ;CHECK:       vpsllq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.sse2.psll.q(<2 x i64> %a0, <2 x i64> %a1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.sse2.psll.q(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i16> @stack_fold_psllw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psllw
  ;CHECK:       vpsllw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse2.psll.w(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse2.psll.w(<8 x i16>, <8 x i16>) nounwind readnone

define <4 x i32> @stack_fold_psrad(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_psrad
  ;CHECK:       vpsrad {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse2.psra.d(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sse2.psra.d(<4 x i32>, <4 x i32>) nounwind readnone

define <8 x i16> @stack_fold_psraw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psraw
  ;CHECK:       vpsraw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse2.psra.w(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse2.psra.w(<8 x i16>, <8 x i16>) nounwind readnone

define <4 x i32> @stack_fold_psrld(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_psrld
  ;CHECK:       vpsrld {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <4 x i32> @llvm.x86.sse2.psrl.d(<4 x i32> %a0, <4 x i32> %a1)
  ret <4 x i32> %2
}
declare <4 x i32> @llvm.x86.sse2.psrl.d(<4 x i32>, <4 x i32>) nounwind readnone

define <2 x i64> @stack_fold_psrlq(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_psrlq
  ;CHECK:       vpsrlq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <2 x i64> @llvm.x86.sse2.psrl.q(<2 x i64> %a0, <2 x i64> %a1)
  ret <2 x i64> %2
}
declare <2 x i64> @llvm.x86.sse2.psrl.q(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i16> @stack_fold_psrlw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psrlw
  ;CHECK:       vpsrlw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse2.psrl.w(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse2.psrl.w(<8 x i16>, <8 x i16>) nounwind readnone

define <16 x i8> @stack_fold_psubb(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_psubb
  ;CHECK:       vpsubb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = sub <16 x i8> %a0, %a1
  ret <16 x i8> %2
}

define <4 x i32> @stack_fold_psubd(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_psubd
  ;CHECK:       vpsubd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = sub <4 x i32> %a0, %a1
  ret <4 x i32> %2
}

; TODO stack_fold_psubq

define <16 x i8> @stack_fold_psubsb(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_psubsb
  ;CHECK:       vpsubsb {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <16 x i8> @llvm.x86.sse2.psubs.b(<16 x i8> %a0, <16 x i8> %a1)
  ret <16 x i8> %2
}
declare <16 x i8> @llvm.x86.sse2.psubs.b(<16 x i8>, <16 x i8>) nounwind readnone

define <8 x i16> @stack_fold_psubsw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psubsw
  ;CHECK:       vpsubsw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = call <8 x i16> @llvm.x86.sse2.psubs.w(<8 x i16> %a0, <8 x i16> %a1)
  ret <8 x i16> %2
}
declare <8 x i16> @llvm.x86.sse2.psubs.w(<8 x i16>, <8 x i16>) nounwind readnone

; TODO stack_fold_psubusb
declare <16 x i8> @llvm.x86.sse2.psubus.b(<16 x i8>, <16 x i8>) nounwind readnone

; TODO stack_fold_psubusw
declare <8 x i16> @llvm.x86.sse2.psubus.w(<8 x i16>, <8 x i16>) nounwind readnone

define <8 x i16> @stack_fold_psubw(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_psubw
  ;CHECK:       vpsubw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = sub <8 x i16> %a0, %a1
  ret <8 x i16> %2
}

; TODO stack_fold_ptest
declare i32 @llvm.x86.sse41.ptestc(<2 x i64>, <2 x i64>) nounwind readnone

define <16 x i8> @stack_fold_punpckhbw(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_punpckhbw
  ;CHECK:       vpunpckhbw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> %a1, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  ret <16 x i8> %2
}

define <4 x i32> @stack_fold_punpckhdq(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_punpckhdq
  ;CHECK:       vpunpckhdq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x i32> %a0, <4 x i32> %a1, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  ret <4 x i32> %2
}

define <2 x i64> @stack_fold_punpckhqdq(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_punpckhqdq
  ;CHECK:       vpunpckhqdq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <2 x i64> %a0, <2 x i64> %a1, <2 x i32> <i32 1, i32 3>
  ret <2 x i64> %2
}

define <8 x i16> @stack_fold_punpckhwd(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_punpckhwd
  ;CHECK:       vpunpckhwd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <8 x i16> %a0, <8 x i16> %a1, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  ret <8 x i16> %2
}

define <16 x i8> @stack_fold_punpcklbw(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_punpcklbw
  ;CHECK:       vpunpcklbw {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <16 x i8> %a0, <16 x i8> %a1, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  ret <16 x i8> %2
}

define <4 x i32> @stack_fold_punpckldq(<4 x i32> %a0, <4 x i32> %a1) {
  ;CHECK-LABEL: stack_fold_punpckldq
  ;CHECK:       vpunpckldq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <4 x i32> %a0, <4 x i32> %a1, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  ret <4 x i32> %2
}

define <2 x i64> @stack_fold_punpcklqdq(<2 x i64> %a0, <2 x i64> %a1) {
  ;CHECK-LABEL: stack_fold_punpcklqdq
  ;CHECK:       vpunpcklqdq {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <2 x i64> %a0, <2 x i64> %a1, <2 x i32> <i32 0, i32 2>
  ret <2 x i64> %2
}

define <8 x i16> @stack_fold_punpcklwd(<8 x i16> %a0, <8 x i16> %a1) {
  ;CHECK-LABEL: stack_fold_punpcklwd
  ;CHECK:       vpunpcklwd {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = shufflevector <8 x i16> %a0, <8 x i16> %a1, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ret <8 x i16> %2
}

define <16 x i8> @stack_fold_pxor(<16 x i8> %a0, <16 x i8> %a1) {
  ;CHECK-LABEL: stack_fold_pxor
  ;CHECK:       vpxor {{-?[0-9]*}}(%rsp), {{%xmm[0-9][0-9]*}}, {{%xmm[0-9][0-9]*}} {{.*#+}} 16-byte Folded Reload
  %1 = tail call <2 x i64> asm sideeffect "nop", "=x,~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{flags}"()
  %2 = xor <16 x i8> %a0, %a1
  ; add forces execution domain
  %3 = add <16 x i8> %2, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  ret <16 x i8> %3
}
