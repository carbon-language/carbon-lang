// RUN: llvm-mc -filetype=obj -triple i386-pc-linux-gnu %s -o - | elf-dump | FileCheck  %s

// Test that we produce the correct relocation types and that the relocation
// to .Lfoo uses the symbol and not the section.

// Section 3 is bss
// CHECK:      # Section 3
// CHECK-NEXT: (('sh_name', 13) # '.bss'

// CHECK:      # Symbol 1
// CHECK-NEXT: (('st_name', 5) # '.Lfoo'

// Symbol 6 is section 3
// CHECK:      # Symbol 6
// CHECK-NEXT: (('st_name', 0) # ''
// CHECK-NEXT:  ('st_value', 0)
// CHECK-NEXT:  ('st_size', 0)
// CHECK-NEXT:  ('st_bind', 0)
// CHECK-NEXT:  ('st_type', 3)
// CHECK-NEXT:  ('st_other', 0)
// CHECK-NEXT:  ('st_shndx', 3)

// CHECK:      # Relocation 0
// CHECK-NEXT: (('r_offset', 2)
// CHECK-NEXT:  ('r_sym', 1)
// CHECK-NEXT:  ('r_type', 9)
// CHECK-NEXT: ),
// CHECK-NEXT:  # Relocation 1
// CHECK-NEXT: (('r_offset',
// CHECK-NEXT:  ('r_sym',
// CHECK-NEXT:  ('r_type', 4)
// CHECK-NEXT: ),
// CHECK-NEXT:  # Relocation 2
// CHECK-NEXT: (('r_offset',
// CHECK-NEXT:  ('r_sym',
// CHECK-NEXT:  ('r_type', 10)
// CHECK-NEXT: ),

// Relocation 3 (bar3@GOTOFF) is done with symbol 6 (bss)
// CHECK-NEXT:  # Relocation 3
// CHECK-NEXT: (('r_offset',
// CHECK-NEXT:  ('r_sym', 6
// CHECK-NEXT:  ('r_type',
// CHECK-NEXT: ),

// Relocation 4 (bar2@GOT) is of type R_386_GOT32
// CHECK-NEXT:  # Relocation 4
// CHECK-NEXT: (('r_offset',
// CHECK-NEXT:  ('r_sym',
// CHECK-NEXT:  ('r_type', 3
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

        .section	.rodata.str1.16,"aMS",@progbits,1
.Lfoo:
	.asciz	 "bool llvm::llvm_start_multithreaded()"
