; This tests for the basic implementation of PPCMachObjectWriter.cpp,
; which is responsible for writing mach-o relocation entries for (PIC)
; PowerPC objects.

; RUN: llvm-mc -filetype=obj -relocation-model=pic -mcpu=g4 -triple=powerpc-apple-darwin8 %s -o - | llvm-readobj -relocations | FileCheck -check-prefix=DARWIN-G4-DUMP %s

	.machine ppc7400
	.section	__TEXT,__textcoal_nt,coalesced,pure_instructions
	.section	__TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
	.align	4
_main:                                  ; @main
; BB#0:                                 ; %entry
	mflr r0
	stw r31, -4(r1)
	stw r0, 8(r1)
	stwu r1, -80(r1)
	bl L0$pb
L0$pb:
	mr r31, r1
	li r5, 0
	mflr 2
	stw r3, 68(r31)
	stw r5, 72(r31)
	stw r4, 64(r31)
	addis r2, r2, ha16(L_.str-L0$pb)
	la r3, lo16(L_.str-L0$pb)(r2)
	bl L_puts$stub
	li r3, 0
	addi r1, r1, 80
	lwz r0, 8(r1)
	lwz r31, -4(r1)
	mtlr r0
	blr

	.section	__TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align	4
L_puts$stub:
	.indirect_symbol	_puts
	mflr r0
	bcl 20, 31, L_puts$stub$tmp
L_puts$stub$tmp:
	mflr r11
	addis r11, r11, ha16(L_puts$lazy_ptr-L_puts$stub$tmp)
	mtlr r0
	lwzu r12, lo16(L_puts$lazy_ptr-L_puts$stub$tmp)(r11)
	mtctr r12
	bctr
	.section	__DATA,__la_symbol_ptr,lazy_symbol_pointers
L_puts$lazy_ptr:
	.indirect_symbol	_puts
	.long	dyld_stub_binding_helper

.subsections_via_symbols
	.section	__TEXT,__cstring,cstring_literals
L_.str:                                 ; @.str
	.asciz	 "Hello, world!"

; DARWIN-G4-DUMP:Format: Mach-O 32-bit ppc
; DARWIN-G4-DUMP:Arch: powerpc
; DARWIN-G4-DUMP:AddressSize: 32bit
; DARWIN-G4-DUMP:Relocations [
; DARWIN-G4-DUMP:  Section __text {
; DARWIN-G4-DUMP:    0x34 1 2 0 PPC_RELOC_BR24 0 0x3
; DARWIN-G4-DUMP:    0x30 0 2 n/a PPC_RELOC_LO16_SECTDIFF 1 0x74
; DARWIN-G4-DUMP:    0x0 0 2 n/a PPC_RELOC_PAIR 1 0x14
; DARWIN-G4-DUMP:    0x2C 0 2 n/a PPC_RELOC_HA16_SECTDIFF 1 0x74
; DARWIN-G4-DUMP:    0x60 0 2 n/a PPC_RELOC_PAIR 1 0x14
; DARWIN-G4-DUMP:  }
; DARWIN-G4-DUMP:  Section __picsymbolstub1 {
; DARWIN-G4-DUMP:    0x14 0 2 n/a PPC_RELOC_LO16_SECTDIFF 1 0x70
; DARWIN-G4-DUMP:    0x0 0 2 n/a PPC_RELOC_PAIR 1 0x58
; DARWIN-G4-DUMP:    0xC 0 2 n/a PPC_RELOC_HA16_SECTDIFF 1 0x70
; DARWIN-G4-DUMP:    0x18 0 2 n/a PPC_RELOC_PAIR 1 0x58
; DARWIN-G4-DUMP:  }
; DARWIN-G4-DUMP:  Section __la_symbol_ptr {
; DARWIN-G4-DUMP:    0x0 0 2 1 PPC_RELOC_VANILLA 0 dyld_stub_binding_helper
; DARWIN-G4-DUMP:  }
; DARWIN-G4-DUMP:]
