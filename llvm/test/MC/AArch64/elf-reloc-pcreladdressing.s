// RUN: llvm-mc -arch=aarch64 -filetype=obj %s -o - | \
// RUN:   elf-dump | FileCheck -check-prefix=OBJ %s

        adr x2, some_label
        adrp x5, some_label

        adrp x5, :got:some_label
        ldr x0, [x5, #:got_lo12:some_label]
// OBJ: .rela.text

// OBJ: 'r_offset', 0x0000000000000000
// OBJ-NEXT:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x00000112

// OBJ: 'r_offset', 0x0000000000000004
// OBJ-NEXT:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x00000113

// OBJ: 'r_offset', 0x0000000000000008
// OBJ-NEXT:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x00000137

// OBJ: 'r_offset', 0x000000000000000c
// OBJ-NEXT:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x00000138

// OBJ: .symtab
// OBJ: Symbol 5
// OBJ-NEXT: some_label