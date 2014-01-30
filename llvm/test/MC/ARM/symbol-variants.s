@ RUN: llvm-mc < %s -triple armv7-none-linux-gnueabi -filetype=obj  | llvm-objdump -triple armv7-none-linux-gnueabi -r - | FileCheck %s --check-prefix=CHECK --check-prefix=ARM
@ RUN: llvm-mc < %s -triple thumbv7-none-linux-gnueabi -filetype=obj  | llvm-objdump -triple thumbv7-none-linux-gnueabi -r - | FileCheck %s --check-prefix=CHECK --check-prefix=THUMB

@ CHECK-LABEL: RELOCATION RECORDS FOR [.rel.text]

@ empty
.word f00
.word f01
@CHECK: 0 R_ARM_ABS32 f00
@CHECK: 4 R_ARM_ABS32 f01

@ none
.word f02(NONE)
.word f03(none)
@CHECK: 8 R_ARM_NONE f02
@CHECK: 12 R_ARM_NONE f03

@ plt
bl f04(PLT)
bl f05(plt)
@ARM: 16 R_ARM_PLT32 f04
@ARM: 20 R_ARM_PLT32 f05
@THUMB: 16 R_ARM_THM_CALL f04
@THUMB: 20 R_ARM_THM_CALL f05

@ got
.word f06(GOT)
.word f07(got)
@CHECK: 24 R_ARM_GOT_BREL f06
@CHECK: 28 R_ARM_GOT_BREL f07

@ gotoff
.word f08(GOTOFF)
.word f09(gotoff)
@CHECK: 32 R_ARM_GOTOFF32 f08
@CHECK: 36 R_ARM_GOTOFF32 f09

@ tpoff
.word f10(TPOFF)
.word f11(tpoff)
@CHECK: 40 R_ARM_TLS_LE32 f10
@CHECK: 44 R_ARM_TLS_LE32 f11

@ tlsgd
.word f12(TLSGD)
.word f13(tlsgd)
@CHECK: 48 R_ARM_TLS_GD32 f12
@CHECK: 52 R_ARM_TLS_GD32 f13

@ target1
.word f14(TARGET1)
.word f15(target1)
@CHECK: 56 R_ARM_TARGET1 f14
@CHECK: 60 R_ARM_TARGET1 f15

@ target2
.word f16(TARGET2)
.word f17(target2)
@CHECK: 64 R_ARM_TARGET2 f16
@CHECK: 68 R_ARM_TARGET2 f17

@ prel31
.word f18(PREL31)
.word f19(prel31)
@CHECK: 72 R_ARM_PREL31 f18
@CHECK: 76 R_ARM_PREL31 f19

@ tlsldo
.word f20(TLSLDO)
.word f21(tlsldo)
@CHECK: 80 R_ARM_TLS_LDO32 f20
@CHECK: 84 R_ARM_TLS_LDO32 f21

@ tlscall
.word f22(TLSCALL)
.word f23(tlscall)
@ CHECK: 88 R_ARM_TLS_CALL f22
@ CHECK: 92 R_ARM_TLS_CALL f23

@ tlsdesc
.word f24(TLSDESC)
.word f25(tlsdesc)
@ CHECK: 96 R_ARM_TLS_GOTDESC f24
@ CHECK: 100 R_ARM_TLS_GOTDESC f25

