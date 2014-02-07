; RUN: llc < %s -march=sparcv9 -filetype=obj --relocation-model=static | llvm-readobj -r | FileCheck %s --check-prefix=CHECK-ABS
; RUN: llc < %s -march=sparcv9 -filetype=obj --relocation-model=pic    | llvm-readobj -r | FileCheck %s --check-prefix=CHECK-PIC

;CHECK-ABS: Relocations [
;CHECK-ABS:    0x{{[0-9,A-F]+}} R_SPARC_H44 AGlobalVar 0x0
;CHECK-ABS:    0x{{[0-9,A-F]+}} R_SPARC_M44 AGlobalVar 0x0
;CHECK-ABS:    0x{{[0-9,A-F]+}} R_SPARC_L44 AGlobalVar 0x0
;CHECK-ABS:    0x{{[0-9,A-F]+}} R_SPARC_WDISP30 bar 0x0
;CHECK-ABS:]

; CHECK-PIC: Relocations [
; CHECK-PIC:    0x{{[0-9,A-F]+}} R_SPARC_PC22 _GLOBAL_OFFSET_TABLE_ 0x4
; CHECK-PIC:    0x{{[0-9,A-F]+}} R_SPARC_PC10 _GLOBAL_OFFSET_TABLE_ 0x8
; CHECK-PIC:    0x{{[0-9,A-F]+}} R_SPARC_GOT22 AGlobalVar 0x0
; CHECK-PIC:    0x{{[0-9,A-F]+}} R_SPARC_GOT10 AGlobalVar 0x0
; CHECK-PIC:    0x{{[0-9,A-F]+}} R_SPARC_WPLT30 bar 0x0
; CHECK-PIC: ]


@AGlobalVar = global i64 0, align 8

; CHECK-ASM: sethi %h44(AGlobalVar), [[R:%[goli][0-7]]]
; CHECK-ASM: add   [[R]], %m44(AGlobalVar), [[R]]
define i64 @foo(i64 %a) {
entry:
  %0 = load i64* @AGlobalVar, align 4
  %1 = add i64 %a, %0
  %2 = call i64 @bar(i64 %1)
  ret i64 %2
}


declare i64 @bar(i64)
