; RUN: llc < %s -march=sparc   -relocation-model=static -code-model=small  | FileCheck --check-prefix=abs32 %s
; RUN: llc < %s -march=sparcv9 -relocation-model=static -code-model=small  | FileCheck --check-prefix=abs32 %s
; RUN: llc < %s -march=sparc   -relocation-model=pic    -code-model=medium | FileCheck --check-prefix=v8pic32 %s

@G = external global i8

define zeroext i8 @loadG() {
  %tmp = load i8* @G
  ret i8 %tmp
}

; abs32: loadG
; abs32: sethi %hi(G), %[[R:[gilo][0-7]]]
; abs32: ldub [%[[R]]+%lo(G)], %i0
; abs32: jmp %i7+8

; v8pic32: loadG
; v8pic32: _GLOBAL_OFFSET_TABLE_
; v8pic32: sethi %hi(G), %[[R1:[gilo][0-7]]]
; v8pic32: add %[[R1]], %lo(G), %[[Goffs:[gilo][0-7]]]
; v8pic32: ld [%[[GOT:[gilo][0-7]]]+%[[Goffs]]], %[[Gaddr:[gilo][0-7]]]
; v8pic32: ldub [%[[Gaddr]]], %i0
; v8pic32: jmp %i7+8
