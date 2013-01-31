// RUN: llvm-mc -arch=aarch64 -filetype=obj %s -o - | \
// RUN:   elf-dump | FileCheck -check-prefix=OBJ %s

        ldr x0, some_label
        ldr w3, some_label
        ldrsw x9, some_label
        prfm pldl3keep, some_label
// OBJ: .rela.text

// OBJ: 'r_offset', 0x0000000000000000
// OBJ:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x00000111

// OBJ: 'r_offset', 0x0000000000000004
// OBJ:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x00000111

// OBJ: 'r_offset', 0x0000000000000008
// OBJ:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x00000111

// OBJ: 'r_offset', 0x000000000000000c
// OBJ:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x00000111

// OBJ: .symtab
// OBJ: Symbol 5
// OBJ-NEXT: some_label