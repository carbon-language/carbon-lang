; This tests for the basic implementation of PPCMachObjectWriter.cpp,
; which is responsible for writing mach-o relocation entries for (PIC)
; PowerPC objects.
; NOTE: Darwin PPC asm syntax is not yet supported by PPCAsmParser,
; so this test case uses ELF PPC asm syntax to produce a mach-o object.
; Once PPCAsmParser supports darwin asm syntax, this test case should
; be updated accordingly.  

; RUN: llvm-mc -filetype=obj -relocation-model=pic -mcpu=g4 -triple=powerpc-apple-darwin8 %s -o - | llvm-readobj -relocations | FileCheck -check-prefix=DARWIN-G4-DUMP %s

;	.machine ppc7400
	.section	__TEXT,__textcoal_nt,coalesced,pure_instructions
	.section	__TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
	.align	4
_main:                                  ; @main
; BB#0:                                 ; %entry
	mflr 0
	stw 31, -4(1)
	stw 0, 8(1)
	stwu 1, -80(1)
	bl L0$pb
L0$pb:
	mr 31, 1
	li 5, 0
	mflr 2
	stw 3, 68(31)
	stw 5, 72(31)
	stw 4, 64(31)
	addis 2, 2, (L_.str-L0$pb)@ha
	la 3, (L_.str-L0$pb)@l(2)
	bl L_puts$stub
	li 3, 0
	addi 1, 1, 80
	lwz 0, 8(1)
	lwz 31, -4(1)
	mtlr 0
	blr

	.section	__TEXT,__picsymbolstub1,symbol_stubs,pure_instructions,32
	.align	4
L_puts$stub:
	.indirect_symbol	_puts
	mflr 0
	bcl 20, 31, L_puts$stub$tmp
L_puts$stub$tmp:
	mflr 11
	addis 11, 11, (L_puts$lazy_ptr-L_puts$stub$tmp)@ha
	mtlr 0
	lwzu 12, (L_puts$lazy_ptr-L_puts$stub$tmp)@l(11)
	mtctr 12
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
; DARWIN-G4-DUMP:    0x34 1 2 0 PPC_RELOC_BR24 0 -
; DARWIN-G4-DUMP:    0x30 0 2 n/a PPC_RELOC_LO16_SECTDIFF 1 _main
; DARWIN-G4-DUMP:    0x0 0 2 n/a PPC_RELOC_PAIR 1 _main
; DARWIN-G4-DUMP:    0x2C 0 2 n/a PPC_RELOC_HA16_SECTDIFF 1 _main
; DARWIN-G4-DUMP:    0x60 0 2 n/a PPC_RELOC_PAIR 1 _main
; DARWIN-G4-DUMP:  }
; DARWIN-G4-DUMP:  Section __picsymbolstub1 {
; DARWIN-G4-DUMP:    0x14 0 2 n/a PPC_RELOC_LO16_SECTDIFF 1 _main
; DARWIN-G4-DUMP:    0x0 0 2 n/a PPC_RELOC_PAIR 1 _main
; DARWIN-G4-DUMP:    0xC 0 2 n/a PPC_RELOC_HA16_SECTDIFF 1 _main
; DARWIN-G4-DUMP:    0x18 0 2 n/a PPC_RELOC_PAIR 1 _main
; DARWIN-G4-DUMP:  }
; DARWIN-G4-DUMP:  Section __la_symbol_ptr {
; DARWIN-G4-DUMP:    0x0 0 2 1 PPC_RELOC_VANILLA 0 dyld_stub_binding_helper
; DARWIN-G4-DUMP:  }
; DARWIN-G4-DUMP:]
