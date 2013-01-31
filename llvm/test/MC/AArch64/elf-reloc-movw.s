// RUN: llvm-mc -arch=aarch64 -filetype=obj %s -o - | \
// RUN:   elf-dump | FileCheck -check-prefix=OBJ %s

        movz x0, #:abs_g0:some_label
        movk x0, #:abs_g0_nc:some_label

        movz x3, #:abs_g1:some_label
        movk x5, #:abs_g1_nc:some_label

        movz x3, #:abs_g2:some_label
        movk x5, #:abs_g2_nc:some_label

        movz x7, #:abs_g3:some_label
        movk x11, #:abs_g3:some_label

        movz x13, #:abs_g0_s:some_label
        movn x17, #:abs_g0_s:some_label

        movz x19, #:abs_g1_s:some_label
        movn x19, #:abs_g1_s:some_label

        movz x19, #:abs_g2_s:some_label
        movn x19, #:abs_g2_s:some_label
// OBJ: .rela.text

// :abs_g0: => R_AARCH64_MOVW_UABS_G0
// OBJ: 'r_offset', 0x0000000000000000
// OBJ:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x00000107

// :abs_g0_nc: => R_AARCH64_MOVW_UABS_G0_NC
// OBJ: 'r_offset', 0x0000000000000004
// OBJ:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x00000108

// :abs_g1: => R_AARCH64_MOVW_UABS_G1
// OBJ: 'r_offset', 0x0000000000000008
// OBJ:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x00000109

// :abs_g1_nc: => R_AARCH64_MOVW_UABS_G1_NC
// OBJ: 'r_offset', 0x000000000000000c
// OBJ:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x0000010a

// :abs_g2: => R_AARCH64_MOVW_UABS_G2
// OBJ: 'r_offset', 0x0000000000000010
// OBJ:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x0000010b

// :abs_g2_nc: => R_AARCH64_MOVW_UABS_G2_NC
// OBJ: 'r_offset', 0x0000000000000014
// OBJ:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x0000010c

// :abs_g3: => R_AARCH64_MOVW_UABS_G3
// OBJ: 'r_offset', 0x0000000000000018
// OBJ:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x0000010d

// :abs_g3: => R_AARCH64_MOVW_UABS_G3
// OBJ: 'r_offset', 0x000000000000001c
// OBJ:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x0000010d

// :abs_g0_s: => R_AARCH64_MOVW_SABS_G0
// OBJ: 'r_offset', 0x0000000000000020
// OBJ:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x0000010e

// :abs_g0_s: => R_AARCH64_MOVW_SABS_G0
// OBJ: 'r_offset', 0x0000000000000024
// OBJ:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x0000010e

// :abs_g1_s: => R_AARCH64_MOVW_SABS_G1
// OBJ: 'r_offset', 0x0000000000000028
// OBJ:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x0000010f

// :abs_g1_s: => R_AARCH64_MOVW_SABS_G1
// OBJ: 'r_offset', 0x000000000000002c
// OBJ:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x0000010f

// :abs_g2_s: => R_AARCH64_MOVW_SABS_G2
// OBJ: 'r_offset', 0x0000000000000030
// OBJ:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x00000110

// :abs_g2_s: => R_AARCH64_MOVW_SABS_G2
// OBJ: 'r_offset', 0x0000000000000034
// OBJ:  'r_sym', 0x00000005
// OBJ-NEXT: 'r_type', 0x00000110

// OBJ: .symtab
// OBJ: Symbol 5
// OBJ-NEXT: some_label
