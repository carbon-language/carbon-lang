; RUN: llc < %s -march=sparc   -relocation-model=static -code-model=small  | FileCheck --check-prefix=abs32 %s
; RUN: llc < %s -march=sparcv9 -relocation-model=static -code-model=small  | FileCheck --check-prefix=abs32 %s
; RUN: llc < %s -march=sparc   -relocation-model=pic    -code-model=medium | FileCheck --check-prefix=v8pic32 %s
; RUN: llc < %s -march=sparcv9 -relocation-model=pic    -code-model=medium | FileCheck --check-prefix=v9pic32 %s

define float @floatCP() {
entry:
  ret float 1.000000e+00
}

; abs32: floatCP
; abs32: sethi %hi(.LCPI0_0), %[[R:[gilo][0-7]]]
; abs32: ld [%[[R]]+%lo(.LCPI0_0)], %f
; abs32: jmp %i7+8

; v8pic32: floatCP
; v8pic32: _GLOBAL_OFFSET_TABLE_
; v8pic32: sethi %hi(.LCPI0_0), %[[R1:[gilo][0-7]]]
; v8pic32: add %[[R1]], %lo(.LCPI0_0), %[[Goffs:[gilo][0-7]]]
; v8pic32: ld [%[[GOT:[gilo][0-7]]]+%[[Goffs]]], %[[Gaddr:[gilo][0-7]]]
; v8pic32: ld [%[[Gaddr]]], %f0
; v8pic32: jmp %i7+8

; v9pic32: floatCP
; v9pic32: _GLOBAL_OFFSET_TABLE_
; v9pic32: sethi %hi(.LCPI0_0), %[[R1:[gilo][0-7]]]
; v9pic32: add %[[R1]], %lo(.LCPI0_0), %[[Goffs:[gilo][0-7]]]
; v9pic32: ldx [%[[GOT:[gilo][0-7]]]+%[[Goffs]]], %[[Gaddr:[gilo][0-7]]]
; v9pic32: ld [%[[Gaddr]]], %f1
; v9pic32: jmp %i7+8
