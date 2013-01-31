// RUN: llvm-mc -arch=aarch64 -filetype=obj %s -o - | \
// RUN:   elf-dump | FileCheck -check-prefix=OBJ %s

        add x2, x3, #:lo12:some_label
// OBJ: .rela.text

// OBJ: 'r_offset', 0x0000000000000000
// OBJ-NEXT:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x00000115

// OBJ: .symtab
// OBJ: Symbol 5
// OBJ-NEXT: some_label