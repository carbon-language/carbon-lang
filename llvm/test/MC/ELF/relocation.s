// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck  %s

// Test that we produce a R_X86_64_32S or R_X86_64_32.

bar:
        movl	$bar, %edx        // R_X86_64_32
        movq	$bar, %rdx        // R_X86_64_32S
        movq	$bar, bar(%rip)   // R_X86_64_32S
        movl	bar, %edx         // R_X86_64_32S
        movq	bar, %rdx         // R_X86_64_32S
.long bar                         // R_X86_64_32


// CHECK: # Relocation 0
// CHECK-NEXT:  (('r_offset', 1)
// CHECK-NEXT:   ('r_sym',
// CHECK-NEXT:   ('r_type', 10)
// CHECK-NEXT:   ('r_addend',

// CHECK: # Relocation 1
// CHECK-NEXT:  (('r_offset', 8)
// CHECK-NEXT:   ('r_sym',
// CHECK-NEXT:   ('r_type', 11)
// CHECK-NEXT:   ('r_addend',

// CHECK: # Relocation 2
// CHECK-NEXT:  (('r_offset', 19)
// CHECK-NEXT:   ('r_sym',
// CHECK-NEXT:   ('r_type', 11)
// CHECK-NEXT:   ('r_addend',

// CHECK: # Relocation 3
// CHECK-NEXT:  (('r_offset', 26)
// CHECK-NEXT:   ('r_sym',
// CHECK-NEXT:   ('r_type', 11)
// CHECK-NEXT:   ('r_addend',

// CHECK: # Relocation 4
// CHECK-NEXT:  (('r_offset', 34)
// CHECK-NEXT:   ('r_sym',
// CHECK-NEXT:   ('r_type', 11)
// CHECK-NEXT:   ('r_addend',

// CHECK: # Relocation 5
// CHECK-NEXT:  (('r_offset', 38)
// CHECK-NEXT:   ('r_sym',
// CHECK-NEXT:   ('r_type', 10)
// CHECK-NEXT:   ('r_addend',
