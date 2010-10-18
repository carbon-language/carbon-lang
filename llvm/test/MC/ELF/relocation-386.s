// RUN: llvm-mc -filetype=obj -triple i386-pc-linux-gnu %s -o - | elf-dump | FileCheck  %s

// Test that we produce the correct relocation types and that the relocation
// to .Lfoo uses the symbol and not the section.


// CHECK:      # Symbol 1
// CHECK-NEXT: (('st_name', 5) # '.Lfoo'

// CHECK:      # Relocation 0
// CHECK-NEXT: (('r_offset', 2)
// CHECK-NEXT:  ('r_sym', 1)
// CHECK-NEXT:  ('r_type', 9)
// CHECK-NEXT: ),
// CHECK-NEXT:  # Relocation 1
// CHECK-NEXT: (('r_offset',
// CHECK-NEXT:  ('r_sym',
// CHECK-NEXT:  ('r_type', 4)

        .text
bar:
	leal	.Lfoo@GOTOFF(%ebx), %eax

        .global bar2
bar2:
	calll	bar2@PLT

        .section	.rodata.str1.16,"aMS",@progbits,1
.Lfoo:
	.asciz	 "bool llvm::llvm_start_multithreaded()"
