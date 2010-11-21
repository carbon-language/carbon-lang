// RUN: llvm-mc -filetype=obj -triple i386-pc-linux-gnu %s -o - | elf-dump | FileCheck  %s

// Test that we produce the correct relocation types and that the relocation
// to .Lfoo uses the symbol and not the section.

// Section 3 is bss
// CHECK:      # Section 0x00000003
// CHECK-NEXT: (('sh_name', 0x0000000d) # '.bss'

// CHECK:      # Symbol 0x00000001
// CHECK-NEXT: (('st_name', 0x00000005) # '.Lfoo'

// Symbol 6 is section 3
// CHECK:      # Symbol 0x00000006
// CHECK-NEXT: (('st_name', 0x00000000) # ''
// CHECK-NEXT:  ('st_value', 0x00000000)
// CHECK-NEXT:  ('st_size', 0x00000000)
// CHECK-NEXT:  ('st_bind', 0x00000000)
// CHECK-NEXT:  ('st_type', 0x00000003)
// CHECK-NEXT:  ('st_other', 0x00000000)
// CHECK-NEXT:  ('st_shndx', 0x00000003)

// CHECK:      # Relocation 0x00000000
// CHECK-NEXT: (('r_offset', 0x00000002)
// CHECK-NEXT:  ('r_sym', 0x00000001)
// CHECK-NEXT:  ('r_type', 0x00000009)
// CHECK-NEXT: ),
// CHECK-NEXT:  # Relocation 0x00000001
// CHECK-NEXT: (('r_offset',
// CHECK-NEXT:  ('r_sym',
// CHECK-NEXT:  ('r_type', 0x00000004)
// CHECK-NEXT: ),
// CHECK-NEXT:  # Relocation 0x00000002
// CHECK-NEXT: (('r_offset',
// CHECK-NEXT:  ('r_sym',
// CHECK-NEXT:  ('r_type', 0x0000000a)
// CHECK-NEXT: ),

// Relocation 3 (bar3@GOTOFF) is done with symbol 6 (bss)
// CHECK-NEXT:  # Relocation 0x00000003
// CHECK-NEXT: (('r_offset',
// CHECK-NEXT:  ('r_sym', 0x00000006
// CHECK-NEXT:  ('r_type',
// CHECK-NEXT: ),

// Relocation 4 (bar2@GOT) is of type R_386_GOT32
// CHECK-NEXT:  # Relocation 0x00000004
// CHECK-NEXT: (('r_offset',
// CHECK-NEXT:  ('r_sym',
// CHECK-NEXT:  ('r_type', 0x00000003
// CHECK-NEXT: ),

// Relocation 5 (foo@TLSGD) is of type R_386_TLS_GD
// CHECK-NEXT: # Relocation 0x00000005
// CHECK-NEXT: (('r_offset', 0x00000020)
// CHECK-NEXT:  ('r_sym', 0x0000000b)
// CHECK-NEXT:  ('r_type', 0x00000012)
// CHECK-NEXT: ),

// Relocation 6 ($foo@TPOFF) is of type R_386_TLS_LE_32
// CHECK-NEXT: # Relocation 0x00000006
// CHECK-NEXT: (('r_offset', 0x00000025)
// CHECK-NEXT:  ('r_sym', 0x0000000b)
// CHECK-NEXT:  ('r_type', 0x00000022)
// CHECK-NEXT: ),

// Relocation 7 (foo@INDNTPOFF) is of type R_386_TLS_IE
// CHECK-NEXT: # Relocation 0x00000007
// CHECK-NEXT: (('r_offset', 0x0000002b)
// CHECK-NEXT:  ('r_sym', 0x0000000b)
// CHECK-NEXT:  ('r_type', 0x0000000f)
// CHECK-NEXT: ),

// Relocation 8 (foo@NTPOFF) is of type R_386_TLS_LE
// CHECK-NEXT: # Relocation 0x00000008
// CHECK-NEXT: (('r_offset', 0x00000031)
// CHECK-NEXT:  ('r_sym', 0x0000000b)
// CHECK-NEXT:  ('r_type', 0x00000011)
// CHECK-NEXT: ),

// Relocation 9 (foo@GOTNTPOFF) is of type R_386_TLS_GOTIE
// CHECK-NEXT: # Relocation 0x00000009
// CHECK-NEXT: (('r_offset', 0x00000037)
// CHECK-NEXT:  ('r_sym', 0x0000000b)
// CHECK-NEXT:  ('r_type', 0x00000010)
// CHECK-NEXT: ),

// Relocation 10 (foo@TLSLDM) is of type R_386_TLS_LDM
// CHECK-NEXT: # Relocation 0x0000000a
// CHECK-NEXT: (('r_offset', 0x0000003d)
// CHECK-NEXT:  ('r_sym', 0x0000000b)
// CHECK-NEXT:  ('r_type', 0x00000013)
// CHECK-NEXT: ),

// Relocation 11 (foo@DTPOFF) is of type R_386_TLS_LDO_32
// CHECK-NEXT: # Relocation 0x0000000b
// CHECK-NEXT: (('r_offset', 0x00000043)
// CHECK-NEXT:  ('r_sym', 0x0000000b)
// CHECK-NEXT:  ('r_type', 0x00000020)
// CHECK-NEXT: ),
// Relocation 12 (calll 4096) is of type R_386_PC32
// CHECK-NEXT: # Relocation 0x0000000c
// CHECK-NEXT: (('r_offset', 0x00000048)
// CHECK-NEXT:  ('r_sym', 0x00000000)
// CHECK-NEXT:  ('r_type', 0x00000002)
// CHECK-NEXT: ),

        .text
bar:
	leal	.Lfoo@GOTOFF(%ebx), %eax

        .global bar2
bar2:
	calll	bar2@PLT
	addl	$_GLOBAL_OFFSET_TABLE_, %ebx
	movb	bar3@GOTOFF(%ebx), %al

	.type	bar3,@object
	.local	bar3
	.comm	bar3,1,1

        movl	bar2j@GOT(%eax), %eax

        leal foo@TLSGD(, %ebx,1), %eax
        movl $foo@TPOFF, %edx
        movl foo@INDNTPOFF, %ecx
        addl foo@NTPOFF(%eax), %eax
        addl foo@GOTNTPOFF(%ebx), %ecx
        leal foo@TLSLDM(%ebx), %eax
        leal foo@DTPOFF(%eax), %edx
        calll 4096

        .section	.rodata.str1.16,"aMS",@progbits,1
.Lfoo:
	.asciz	 "bool llvm::llvm_start_multithreaded()"
