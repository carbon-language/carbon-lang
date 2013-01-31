// RUN: llvm-mc -arch=aarch64 -filetype=obj %s -o - | \
// RUN:   elf-dump | FileCheck -check-prefix=OBJ %s

        ldrb w0, [sp, #:lo12:some_label]
        ldrh w0, [sp, #:lo12:some_label]
        ldr w0, [sp, #:lo12:some_label]
        ldr x0, [sp, #:lo12:some_label]
        str q0, [sp, #:lo12:some_label]

// OBJ: .rela.text

// OBJ: 'r_offset', 0x0000000000000000
// OBJ-NEXT:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x00000116

// OBJ: 'r_offset', 0x0000000000000004
// OBJ-NEXT:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x0000011c

// OBJ: 'r_offset', 0x0000000000000008
// OBJ-NEXT:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x0000011d

// OBJ: 'r_offset', 0x000000000000000c
// OBJ-NEXT:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x0000011e

// OBJ: 'r_offset', 0x0000000000000010
// OBJ-NEXT:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x0000012b

// OBJ: .symtab
// OBJ: Symbol 5
// OBJ-NEXT: some_label
