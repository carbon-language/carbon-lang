// RUN: llvm-mc -arch=aarch64 -filetype=obj %s -o - | \
// RUN:   elf-dump | FileCheck -check-prefix=OBJ %s

        tbz x6, #45, somewhere
        tbnz w3, #15, somewhere
// OBJ: .rela.text

// OBJ: 'r_offset', 0x0000000000000000
// OBJ-NEXT:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x00000117

// OBJ: 'r_offset', 0x0000000000000004
// OBJ-NEXT:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x00000117

// OBJ: .symtab
// OBJ: Symbol 5
// OBJ-NEXT: somewhere
