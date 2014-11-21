; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx | FileCheck %s --check-prefix=SANDYB
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx-i | FileCheck %s --check-prefix=SANDYB
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=btver2 | FileCheck %s --check-prefix=BTVER2
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 | FileCheck %s --check-prefix=HASWELL

; On Sandy Bridge or Ivy Bridge, we should not generate an unaligned 32-byte load
; because that is slower than two 16-byte loads. 
; Other AVX-capable chips don't have that problem.

define <8 x float> @load32bytes(<8 x float>* %Ap) {
  ; CHECK-LABEL: load32bytes

  ; SANDYB: vmovaps
  ; SANDYB: vinsertf128
  ; SANDYB: retq

  ; BTVER2: vmovups
  ; BTVER2: retq

  ; HASWELL: vmovups
  ; HASWELL: retq

  %A = load <8 x float>* %Ap, align 16
  ret <8 x float> %A
}

; On Sandy Bridge or Ivy Bridge, we should not generate an unaligned 32-byte store
; because that is slowerthan two 16-byte stores. 
; Other AVX-capable chips don't have that problem.

define void @store32bytes(<8 x float> %A, <8 x float>* %P) {
  ; CHECK-LABEL: store32bytes

  ; SANDYB: vextractf128
  ; SANDYB: vmovaps
  ; SANDYB: retq

  ; BTVER2: vmovups
  ; BTVER2: retq

  ; HASWELL: vmovups
  ; HASWELL: retq

  store <8 x float> %A, <8 x float>* %P, align 16
  ret void
}
