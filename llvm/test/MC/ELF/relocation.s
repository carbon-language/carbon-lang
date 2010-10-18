// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck  %s

// Test that we produce a R_X86_64_32S or R_X86_64_32.

bar:
        movl	$bar, %edx        // R_X86_64_32
        movq	$bar, %rdx        // R_X86_64_32S
        movq	$bar, bar(%rip)   // R_X86_64_32S
        movl	bar, %edx         // R_X86_64_32S
        movq	bar, %rdx         // R_X86_64_32S
.long bar                         // R_X86_64_32

// CHECK:  # Section 0x1
// CHECK: (('sh_name', 0x1) # '.text'

// CHECK:   # Symbol 0x2
// CHECK: (('st_name', 0x0) # ''
// CHECK:  ('st_bind', 0x0)
// CHECK   ('st_type', 0x3)
// CHECK:  ('st_other', 0x0)
// CHECK:  ('st_shndx', 0x1)

// CHECK: # Relocation 0x0
// CHECK-NEXT:  (('r_offset', 0x1)
// CHECK-NEXT:   ('r_sym', 0x2)
// CHECK-NEXT:   ('r_type', 0xa)
// CHECK-NEXT:   ('r_addend',

// CHECK: # Relocation 0x1
// CHECK-NEXT:  (('r_offset', 0x8)
// CHECK-NEXT:   ('r_sym', 0x2)
// CHECK-NEXT:   ('r_type', 0xb)
// CHECK-NEXT:   ('r_addend',

// CHECK: # Relocation 0x2
// CHECK-NEXT:  (('r_offset', 0x13)
// CHECK-NEXT:   ('r_sym', 0x2)
// CHECK-NEXT:   ('r_type', 0xb)
// CHECK-NEXT:   ('r_addend',

// CHECK: # Relocation 0x3
// CHECK-NEXT:  (('r_offset', 0x1a)
// CHECK-NEXT:   ('r_sym', 0x2)
// CHECK-NEXT:   ('r_type', 0xb)
// CHECK-NEXT:   ('r_addend',

// CHECK: # Relocation 0x4
// CHECK-NEXT:  (('r_offset', 0x22)
// CHECK-NEXT:   ('r_sym', 0x2)
// CHECK-NEXT:   ('r_type', 0xb)
// CHECK-NEXT:   ('r_addend',

// CHECK: # Relocation 0x5
// CHECK-NEXT:  (('r_offset', 0x26)
// CHECK-NEXT:   ('r_sym', 0x2)
// CHECK-NEXT:   ('r_type', 0xa)
// CHECK-NEXT:   ('r_addend',
