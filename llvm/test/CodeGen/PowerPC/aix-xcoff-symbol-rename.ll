;; This file tests how llc handles symbols containing invalid characters on an
;; XCOFF platform.
;; Since symbol name resolution is the same between 32-bit and 64-bit,
;; tests for 64-bit mode are omitted.

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -data-sections=false -xcoff-traceback-table=false < %s | \
; RUN:   FileCheck --check-prefix=ASM %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -data-sections=false -xcoff-traceback-table=false -filetype=obj -o %t.o < %s
; RUN: llvm-objdump -D -r --symbol-description %t.o | \
; RUN:   FileCheck --check-prefix=OBJ %s

; This is f`o
@"f\60o" = global i32 10, align 4

; This is f"o"
@"f\22o\22" = common global i32 0, align 4

; This is f=o
@"f\3do" = internal global i32 0, align 4

define internal i32 @f$o() {
entry:
  %call = call i32 bitcast (i32 (...)* @"f\40o" to i32 ()*)()
  ret i32 %call
}

; This is f&o
define i32 @"f\26o"() {
entry:
  %tmp = call i32 @f$o()
  %tmp1 = load i32, i32* @"f\3do"
  %tmp2 = add i32 %tmp, %tmp1
  ret i32 %tmp2
}

; This is f&_o
define i32 (...)* @"f\26_o"() {
entry:
  ret i32 (...)* @"f\40o"
}

; This is f@o
declare i32 @"f\40o"(...)

; ASM:         .lglobl _Renamed..24f_o[DS] # -- Begin function f$o
; ASM-NEXT:    .rename _Renamed..24f_o[DS],"f$o"
; ASM-NEXT:    .lglobl ._Renamed..24f_o
; ASM-NEXT:    .rename ._Renamed..24f_o,".f$o"
; ASM-NEXT:    .align  4
; ASM-NEXT:    .csect _Renamed..24f_o[DS],2
; ASM-NEXT:    .vbyte  4, ._Renamed..24f_o     # @"f$o"
; ASM-NEXT:    .vbyte  4, TOC[TC0]
; ASM-NEXT:    .vbyte  4, 0
; ASM-NEXT:    .csect .text[PR],2
; ASM-NEXT:  ._Renamed..24f_o:
; ASM:         bl ._Renamed..40f_o[PR]
; ASM-NEXT:    nop
; ASM:         .globl  _Renamed..26f_o[DS] # -- Begin function f&o
; ASM-NEXT:    .rename _Renamed..26f_o[DS],"f&o"
; ASM-NEXT:    .globl  ._Renamed..26f_o
; ASM-NEXT:    .rename ._Renamed..26f_o,".f&o"
; ASM-NEXT:    .align  4
; ASM-NEXT:    .csect _Renamed..26f_o[DS],2
; ASM-NEXT:    .vbyte  4, ._Renamed..26f_o     # @"f&o"
; ASM-NEXT:    .vbyte  4, TOC[TC0]
; ASM-NEXT:    .vbyte  4, 0
; ASM-NEXT:    .csect .text[PR],2
; ASM-NEXT:  ._Renamed..26f_o:
; ASM:         bl ._Renamed..24f_o
; ASM:         .globl  _Renamed..265ff__o[DS] # -- Begin function f&_o
; ASM-NEXT:    .rename _Renamed..265ff__o[DS],"f&_o"
; ASM-NEXT:    .globl  ._Renamed..265ff__o
; ASM-NEXT:    .rename ._Renamed..265ff__o,".f&_o"
; ASM-NEXT:    .align  4
; ASM-NEXT:    .csect _Renamed..265ff__o[DS],2
; ASM-NEXT:    .vbyte  4, ._Renamed..265ff__o  # @"f&_o"
; ASM-NEXT:    .vbyte  4, TOC[TC0]
; ASM-NEXT:    .vbyte  4, 0
; ASM-NEXT:    .csect .text[PR],2
; ASM-NEXT:  ._Renamed..265ff__o:
; ASM:         .csect .data[RW],2
; ASM-NEXT:    .globl  _Renamed..60f_o
; ASM-NEXT:    .rename _Renamed..60f_o,"f`o"
; ASM-NEXT:    .align  2
; ASM-NEXT:  _Renamed..60f_o:
; ASM-NEXT:    .vbyte  4, 10                   # 0xa
; ASM-NEXT:    .comm _Renamed..2222f_o_[RW],4,2
; ASM-NEXT:    .rename _Renamed..2222f_o_[RW],"f""o"""
; ASM-NEXT:    .lcomm  _Renamed..3df_o,4,_Renamed..3df_o[BS],2
; ASM-NEXT:    .rename _Renamed..3df_o[BS],"f=o"
; ASM-NEXT:    .extern ._Renamed..40f_o[PR]
; ASM-NEXT:    .rename ._Renamed..40f_o[PR],".f@o"
; ASM-NEXT:    .extern _Renamed..40f_o[DS]
; ASM-NEXT:    .rename _Renamed..40f_o[DS],"f@o"
; ASM-NEXT:    .toc
; ASM-NEXT:  L..C0:
; ASM-NEXT:    .tc _Renamed..3df_o[TC],_Renamed..3df_o[BS]
; ASM-NEXT:    .rename _Renamed..3df_o[TC],"f=o"
; ASM-NEXT:  L..C1:
; ASM-NEXT:    .tc _Renamed..40f_o[TC],_Renamed..40f_o[DS]
; ASM-NEXT:    .rename _Renamed..40f_o[TC],"f@o"

; OBJ:       Disassembly of section .text:
; OBJ-EMPTY:
; OBJ-NEXT:  00000000 (idx: 7) .f$o:
; OBJ-NEXT:         0: 7c 08 02 a6   mflr 0
; OBJ-NEXT:         4: 90 01 00 08   stw 0, 8(1)
; OBJ-NEXT:         8: 94 21 ff c0   stwu 1, -64(1)
; OBJ-NEXT:         c: 4b ff ff f5   bl 0x0
; OBJ-NEXT:                          0000000c:  R_RBR        (idx: 1) .f@o[PR]
; OBJ-NEXT:        10: 60 00 00 00   nop
; OBJ-NEXT:        14: 38 21 00 40   addi 1, 1, 64
; OBJ-NEXT:        18: 80 01 00 08   lwz 0, 8(1)
; OBJ-NEXT:        1c: 7c 08 03 a6   mtlr 0
; OBJ-NEXT:        20: 4e 80 00 20   blr
; OBJ-NEXT:        24: 60 00 00 00   nop
; OBJ-NEXT:        28: 60 00 00 00   nop
; OBJ-NEXT:        2c: 60 00 00 00   nop
; OBJ-EMPTY:
; OBJ-NEXT:  00000030 (idx: 9) .f&o:
; OBJ-NEXT:        30: 7c 08 02 a6   mflr 0
; OBJ-NEXT:        34: 90 01 00 08   stw 0, 8(1)
; OBJ-NEXT:        38: 94 21 ff c0   stwu 1, -64(1)
; OBJ-NEXT:        3c: 4b ff ff c5   bl 0x0
; OBJ-NEXT:        40: 80 82 00 00   lwz 4, 0(2)
; OBJ-NEXT:                          00000042:  R_TOC        (idx: 25) f=o[TC]
; OBJ-NEXT:        44: 80 84 00 00   lwz 4, 0(4)
; OBJ-NEXT:        48: 7c 63 22 14   add 3, 3, 4
; OBJ-NEXT:        4c: 38 21 00 40   addi 1, 1, 64
; OBJ-NEXT:        50: 80 01 00 08   lwz 0, 8(1)
; OBJ-NEXT:        54: 7c 08 03 a6   mtlr 0
; OBJ-NEXT:        58: 4e 80 00 20   blr
; OBJ-NEXT:        5c: 60 00 00 00   nop
; OBJ-EMPTY:
; OBJ-NEXT:  00000060 (idx: 11) .f&_o:
; OBJ-NEXT:        60: 80 62 00 04   lwz 3, 4(2)
; OBJ-NEXT:                          00000062:  R_TOC        (idx: 27) f@o[TC]
; OBJ-NEXT:        64: 4e 80 00 20   blr
; OBJ-EMPTY:
; OBJ-NEXT:  Disassembly of section .data:
; OBJ-EMPTY:
; OBJ-NEXT:  00000068 (idx: 15) f`o:
; OBJ-NEXT:        68: 00 00 00 0a   <unknown>
; OBJ-EMPTY:
; OBJ-NEXT:  0000006c (idx: 17) f$o[DS]:
; OBJ-NEXT:        6c: 00 00 00 00   <unknown>
; OBJ-NEXT:                          0000006c:  R_POS        (idx: 7) .f$o
; OBJ-NEXT:        70: 00 00 00 90   <unknown>
; OBJ-NEXT:                          00000070:  R_POS        (idx: 23) TOC[TC0]
; OBJ-NEXT:        74: 00 00 00 00   <unknown>
; OBJ-EMPTY:
; OBJ-NEXT:  00000078 (idx: 19) f&o[DS]:
; OBJ-NEXT:        78: 00 00 00 30   <unknown>
; OBJ-NEXT:                          00000078:  R_POS        (idx: 9) .f&o
; OBJ-NEXT:        7c: 00 00 00 90   <unknown>
; OBJ-NEXT:                          0000007c:  R_POS        (idx: 23) TOC[TC0]
; OBJ-NEXT:        80: 00 00 00 00   <unknown>
; OBJ-EMPTY:
; OBJ-NEXT:  00000084 (idx: 21) f&_o[DS]:
; OBJ-NEXT:        84: 00 00 00 60   <unknown>
; OBJ-NEXT:                          00000084:  R_POS        (idx: 11) .f&_o
; OBJ-NEXT:        88: 00 00 00 90   <unknown>
; OBJ-NEXT:                          00000088:  R_POS        (idx: 23) TOC[TC0]
; OBJ-NEXT:        8c: 00 00 00 00   <unknown>
; OBJ-EMPTY:
; OBJ-NEXT:  00000090 (idx: 25) f=o[TC]:
; OBJ-NEXT:        90: 00 00 00 9c   <unknown>
; OBJ-NEXT:                          00000090:  R_POS        (idx: 31) f=o[BS]
; OBJ-EMPTY:
; OBJ-NEXT:  00000094 (idx: 27) f@o[TC]:
; OBJ-NEXT:        94: 00 00 00 00   <unknown>
; OBJ-NEXT:                          00000094:  R_POS        (idx: 3) f@o[DS]
; OBJ-EMPTY:
; OBJ-NEXT:  Disassembly of section .bss:
; OBJ-EMPTY:
; OBJ-NEXT:  00000098 (idx: 29) f"o"[RW]:
; OBJ-NEXT:  ...
; OBJ-EMPTY:
; OBJ-NEXT:  0000009c (idx: 31) f=o[BS]:
; OBJ-NEXT:  ...
