// RUN: llvm-mc -arch=aarch64 -filetype=obj %s -o - | \
// RUN:   elf-dump | FileCheck -check-prefix=OBJ %s

        b somewhere
        bl somewhere
// OBJ: .rela.text

// OBJ: 'r_offset', 0x0000000000000000
// OBJ-NEXT:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x0000011a

// OBJ: 'r_offset', 0x0000000000000004
// OBJ-NEXT:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x0000011b

// OBJ: .symtab
// OBJ: Symbol 5
// OBJ-NEXT: somewhere