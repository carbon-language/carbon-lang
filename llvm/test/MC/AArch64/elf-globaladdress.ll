;; RUN: llc -mtriple=aarch64-none-linux-gnu -filetype=obj %s -o - | \
;; RUN:   elf-dump | FileCheck -check-prefix=OBJ %s

; Also take it on a round-trip through llvm-mc to stretch assembly-parsing's legs:
;; RUN: llc -mtriple=aarch64-none-linux-gnu %s -o - | \
;; RUN:     llvm-mc -arch=aarch64 -filetype=obj -o - | \
;; RUN:     elf-dump | FileCheck -check-prefix=OBJ %s

@var8 = global i8 0
@var16 = global i16 0
@var32 = global i32 0
@var64 = global i64 0

define void @loadstore() {
    %val8 = load i8* @var8
    store volatile i8 %val8, i8* @var8

    %val16 = load i16* @var16
    store volatile i16 %val16, i16* @var16

    %val32 = load i32* @var32
    store volatile i32 %val32, i32* @var32

    %val64 = load i64* @var64
    store volatile i64 %val64, i64* @var64

    ret void
}

@globaddr = global i64* null

define void @address() {
    store i64* @var64, i64** @globaddr
    ret void
}

; Check we're using EM_AARCH64
; OBJ: 'e_machine', 0x00

; OBJ: .rela.text

; var8
; R_AARCH64_ADR_PREL_PG_HI21 against var8
; OBJ: 'r_sym', 0x0000000f
; OBJ-NEXT: 'r_type', 0x00000113

; R_AARCH64_LDST8_ABS_LO12_NC against var8
; OBJ: 'r_sym', 0x0000000f
; OBJ-NEXT: 'r_type', 0x00000116


; var16
; R_AARCH64_ADR_PREL_PG_HI21 against var16
; OBJ: 'r_sym', 0x0000000c
; OBJ-NEXT: 'r_type', 0x00000113

; R_AARCH64_LDST16_ABS_LO12_NC against var16
; OBJ: 'r_sym', 0x0000000c
; OBJ-NEXT: 'r_type', 0x0000011c


; var32
; R_AARCH64_ADR_PREL_PG_HI21 against var32
; OBJ: 'r_sym', 0x0000000d
; OBJ-NEXT: 'r_type', 0x00000113

; R_AARCH64_LDST32_ABS_LO12_NC against var32
; OBJ: 'r_sym', 0x0000000d
; OBJ-NEXT: 'r_type', 0x0000011d


; var64
; R_AARCH64_ADR_PREL_PG_HI21 against var64
; OBJ: 'r_sym', 0x0000000e
; OBJ-NEXT: 'r_type', 0x00000113

; R_AARCH64_LDST64_ABS_LO12_NC against var64
; OBJ: 'r_sym', 0x0000000e
; OBJ-NEXT: 'r_type', 0x0000011e

; This is on the store, so not really important, but it stops the next
; match working.
; R_AARCH64_LDST64_ABS_LO12_NC against var64
; OBJ: 'r_sym', 0x0000000e
; OBJ-NEXT: 'r_type', 0x0000011e


; Pure address-calculation against var64
; R_AARCH64_ADR_PREL_PG_HI21 against var64
; OBJ: 'r_sym', 0x0000000e
; OBJ-NEXT: 'r_type', 0x00000113

; R_AARCH64_ADD_ABS_LO12_NC against var64
; OBJ: 'r_sym', 0x0000000e
; OBJ-NEXT: 'r_type', 0x00000115


; Make sure the symbols don't move around, otherwise relocation info
; will be wrong:

; OBJ: Symbol 12
; OBJ-NEXT: var16

; OBJ: Symbol 13
; OBJ-NEXT: var32

; OBJ: Symbol 14
; OBJ-NEXT: var64

; OBJ: Symbol 15
; OBJ-NEXT: var8
