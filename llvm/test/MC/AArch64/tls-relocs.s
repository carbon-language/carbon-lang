// RUN: llvm-mc -arch=aarch64 -show-encoding < %s | FileCheck %s
// RUN: llvm-mc -arch=aarch64 -filetype=obj < %s -o %t
// RUN: elf-dump %t | FileCheck --check-prefix=CHECK-ELF %s
// RUN: llvm-objdump -r %t | FileCheck --check-prefix=CHECK-ELF-NAMES %s

// CHECK-ELF:  .rela.text

        // TLS local-dynamic forms
        movz x1, #:dtprel_g2:var
        movn x2, #:dtprel_g2:var
        movz x3, #:dtprel_g2:var
        movn x4, #:dtprel_g2:var
// CHECK: movz    x1, #:dtprel_g2:var     // encoding: [0x01'A',A,0xc0'A',0x92'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_g2:var, kind: fixup_a64_movw_dtprel_g2
// CHECK-NEXT: movn    x2, #:dtprel_g2:var     // encoding: [0x02'A',A,0xc0'A',0x92'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_g2:var, kind: fixup_a64_movw_dtprel_g2
// CHECK-NEXT: movz    x3, #:dtprel_g2:var     // encoding: [0x03'A',A,0xc0'A',0x92'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_g2:var, kind: fixup_a64_movw_dtprel_g2
// CHECK-NEXT: movn    x4, #:dtprel_g2:var     // encoding: [0x04'A',A,0xc0'A',0x92'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_g2:var, kind: fixup_a64_movw_dtprel_g2

// CHECK-ELF: # Relocation 0
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000000)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM:0x[0-9a-f]+]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000020b)
// CHECK-ELF: # Relocation 1
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000004)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000020b)
// CHECK-ELF: # Relocation 2
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000008)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000020b)
// CHECK-ELF: # Relocation 3
// CHECK-ELF-NEXT: (('r_offset', 0x000000000000000c)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000020b)

// CHECK-ELF-NAMES: 0 R_AARCH64_TLSLD_MOVW_DTPREL_G2
// CHECK-ELF-NAMES: 4 R_AARCH64_TLSLD_MOVW_DTPREL_G2
// CHECK-ELF-NAMES: 8 R_AARCH64_TLSLD_MOVW_DTPREL_G2
// CHECK-ELF-NAMES: 12 R_AARCH64_TLSLD_MOVW_DTPREL_G2

        movz x5, #:dtprel_g1:var
        movn x6, #:dtprel_g1:var
        movz w7, #:dtprel_g1:var
        movn w8, #:dtprel_g1:var
// CHECK: movz    x5, #:dtprel_g1:var     // encoding: [0x05'A',A,0xa0'A',0x92'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_g1:var, kind: fixup_a64_movw_dtprel_g1
// CHECK-NEXT: movn    x6, #:dtprel_g1:var     // encoding: [0x06'A',A,0xa0'A',0x92'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_g1:var, kind: fixup_a64_movw_dtprel_g1
// CHECK-NEXT: movz    w7, #:dtprel_g1:var     // encoding: [0x07'A',A,0xa0'A',0x12'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_g1:var, kind: fixup_a64_movw_dtprel_g1
// CHECK-NEXT: movn    w8, #:dtprel_g1:var     // encoding: [0x08'A',A,0xa0'A',0x12'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_g1:var, kind: fixup_a64_movw_dtprel_g1

// CHECK-ELF: # Relocation 4
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000010)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000020c)
// CHECK-ELF: # Relocation 5
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000014)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000020c)
// CHECK-ELF: # Relocation 6
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000018)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000020c)
// CHECK-ELF: # Relocation 7
// CHECK-ELF-NEXT: (('r_offset', 0x000000000000001c)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000020c)

// CHECK-ELF-NAMES: 16 R_AARCH64_TLSLD_MOVW_DTPREL_G1
// CHECK-ELF-NAMES: 20 R_AARCH64_TLSLD_MOVW_DTPREL_G1
// CHECK-ELF-NAMES: 24 R_AARCH64_TLSLD_MOVW_DTPREL_G1
// CHECK-ELF-NAMES: 28 R_AARCH64_TLSLD_MOVW_DTPREL_G1

        movk x9, #:dtprel_g1_nc:var
        movk w10, #:dtprel_g1_nc:var
// CHECK: movk    x9, #:dtprel_g1_nc:var  // encoding: [0x09'A',A,0xa0'A',0xf2'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_g1_nc:var, kind: fixup_a64_movw_dtprel_g1_nc
// CHECK-NEXT: movk    w10, #:dtprel_g1_nc:var // encoding: [0x0a'A',A,0xa0'A',0x72'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_g1_nc:var, kind: fixup_a64_movw_dtprel_g1_nc

// CHECK-ELF: # Relocation 8
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000020)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000020d)
// CHECK-ELF: # Relocation 9
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000024)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000020d)

// CHECK-ELF-NAMES: 32 R_AARCH64_TLSLD_MOVW_DTPREL_G1_NC
// CHECK-ELF-NAMES: 36 R_AARCH64_TLSLD_MOVW_DTPREL_G1_NC

        movz x11, #:dtprel_g0:var
        movn x12, #:dtprel_g0:var
        movz w13, #:dtprel_g0:var
        movn w14, #:dtprel_g0:var
// CHECK: movz    x11, #:dtprel_g0:var    // encoding: [0x0b'A',A,0x80'A',0x92'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_g0:var, kind: fixup_a64_movw_dtprel_g0
// CHECK-NEXT: movn    x12, #:dtprel_g0:var    // encoding: [0x0c'A',A,0x80'A',0x92'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_g0:var, kind: fixup_a64_movw_dtprel_g0
// CHECK-NEXT: movz    w13, #:dtprel_g0:var    // encoding: [0x0d'A',A,0x80'A',0x12'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_g0:var, kind: fixup_a64_movw_dtprel_g0
// CHECK-NEXT: movn    w14, #:dtprel_g0:var    // encoding: [0x0e'A',A,0x80'A',0x12'A']


// CHECK-ELF: # Relocation 10
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000028)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000020e)
// CHECK-ELF: # Relocation 11
// CHECK-ELF-NEXT: (('r_offset', 0x000000000000002c)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000020e)
// CHECK-ELF: # Relocation 12
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000030)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000020e)
// CHECK-ELF: # Relocation 13
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000034)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000020e)

// CHECK-ELF-NAMES: 40 R_AARCH64_TLSLD_MOVW_DTPREL_G0
// CHECK-ELF-NAMES: 44 R_AARCH64_TLSLD_MOVW_DTPREL_G0
// CHECK-ELF-NAMES: 48 R_AARCH64_TLSLD_MOVW_DTPREL_G0
// CHECK-ELF-NAMES: 52 R_AARCH64_TLSLD_MOVW_DTPREL_G0


        movk x15, #:dtprel_g0_nc:var
        movk w16, #:dtprel_g0_nc:var
// CHECK: movk    x15, #:dtprel_g0_nc:var // encoding: [0x0f'A',A,0x80'A',0xf2'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_g0_nc:var, kind: fixup_a64_movw_dtprel_g0_nc
// CHECK-NEXT: movk    w16, #:dtprel_g0_nc:var // encoding: [0x10'A',A,0x80'A',0x72'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_g0_nc:var, kind: fixup_a64_movw_dtprel_g0_nc

// CHECK-ELF: # Relocation 14
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000038)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000020f)
// CHECK-ELF: # Relocation 15
// CHECK-ELF-NEXT: (('r_offset', 0x000000000000003c)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000020f)

// CHECK-ELF-NAMES: 56 R_AARCH64_TLSLD_MOVW_DTPREL_G0_NC
// CHECK-ELF-NAMES: 60 R_AARCH64_TLSLD_MOVW_DTPREL_G0_NC

        add x17, x18, #:dtprel_hi12:var, lsl #12
        add w19, w20, #:dtprel_hi12:var, lsl #12
// CHECK: add     x17, x18, #:dtprel_hi12:var, lsl #12 // encoding: [0x51'A',0x02'A',0x40'A',0x91'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_hi12:var, kind: fixup_a64_add_dtprel_hi12
// CHECK-NEXT: add     w19, w20, #:dtprel_hi12:var, lsl #12 // encoding: [0x93'A',0x02'A',0x40'A',0x11'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_hi12:var, kind: fixup_a64_add_dtprel_hi12

// CHECK-ELF: # Relocation 16
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000040)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000210)
// CHECK-ELF: # Relocation 17
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000044)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000210)

// CHECK-ELF-NAMES: 64 R_AARCH64_TLSLD_ADD_DTPREL_HI12
// CHECK-ELF-NAMES: 68 R_AARCH64_TLSLD_ADD_DTPREL_HI12


        add x21, x22, #:dtprel_lo12:var
        add w23, w24, #:dtprel_lo12:var
// CHECK: add     x21, x22, #:dtprel_lo12:var // encoding: [0xd5'A',0x02'A',A,0x91'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_lo12:var, kind: fixup_a64_add_dtprel_lo12
// CHECK-NEXT: add     w23, w24, #:dtprel_lo12:var // encoding: [0x17'A',0x03'A',A,0x11'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_lo12:var, kind: fixup_a64_add_dtprel_lo12

// CHECK-ELF: # Relocation 18
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000048)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000211)
// CHECK-ELF: # Relocation 19
// CHECK-ELF-NEXT: (('r_offset', 0x000000000000004c)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000211)

// CHECK-ELF-NAMES: 72 R_AARCH64_TLSLD_ADD_DTPREL_LO12
// CHECK-ELF-NAMES: 76 R_AARCH64_TLSLD_ADD_DTPREL_LO12

        add x25, x26, #:dtprel_lo12_nc:var
        add w27, w28, #:dtprel_lo12_nc:var
// CHECK: add     x25, x26, #:dtprel_lo12_nc:var // encoding: [0x59'A',0x03'A',A,0x91'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_lo12_nc:var, kind: fixup_a64_add_dtprel_lo12_nc
// CHECK-NEXT: add     w27, w28, #:dtprel_lo12_nc:var // encoding: [0x9b'A',0x03'A',A,0x11'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_lo12_nc:var, kind: fixup_a64_add_dtprel_lo12_nc

// CHECK-ELF: # Relocation 20
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000050)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000212)
// CHECK-ELF: # Relocation 21
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000054)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000212)

// CHECK-ELF-NAMES: 80 R_AARCH64_TLSLD_ADD_DTPREL_LO12_NC
// CHECK-ELF-NAMES: 84 R_AARCH64_TLSLD_ADD_DTPREL_LO12_NC

        ldrb w29, [x30, #:dtprel_lo12:var]
        ldrsb x29, [x28, #:dtprel_lo12_nc:var]
// CHECK: ldrb    w29, [x30, #:dtprel_lo12:var] // encoding: [0xdd'A',0x03'A',0x40'A',0x39'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_lo12:var, kind: fixup_a64_ldst8_dtprel_lo12
// CHECK-NEXT: ldrsb   x29, [x28, #:dtprel_lo12_nc:var] // encoding: [0x9d'A',0x03'A',0x80'A',0x39'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_lo12_nc:var, kind: fixup_a64_ldst8_dtprel_lo12_nc

// CHECK-ELF: # Relocation 22
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000058)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000213)
// CHECK-ELF: # Relocation 23
// CHECK-ELF-NEXT: (('r_offset', 0x000000000000005c)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000214)

// CHECK-ELF-NAMES: 88 R_AARCH64_TLSLD_LDST8_DTPREL_LO12
// CHECK-ELF-NAMES: 92 R_AARCH64_TLSLD_LDST8_DTPREL_LO12_NC

        strh w27, [x26, #:dtprel_lo12:var]
        ldrsh x25, [x24, #:dtprel_lo12_nc:var]
// CHECK: strh    w27, [x26, #:dtprel_lo12:var] // encoding: [0x5b'A',0x03'A',A,0x79'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_lo12:var, kind: fixup_a64_ldst16_dtprel_lo12
// CHECK-NEXT: ldrsh   x25, [x24, #:dtprel_lo12_nc:var] // encoding: [0x19'A',0x03'A',0x80'A',0x79'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_lo12_nc:var, kind: fixup_a64_ldst16_dtprel_lo12_n

// CHECK-ELF: # Relocation 24
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000060)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000215)
// CHECK-ELF: # Relocation 25
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000064)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000216)

// CHECK-ELF-NAMES: 96 R_AARCH64_TLSLD_LDST16_DTPREL_LO12
// CHECK-ELF-NAMES: 100 R_AARCH64_TLSLD_LDST16_DTPREL_LO12_NC

        ldr w23, [x22, #:dtprel_lo12:var]
        ldrsw x21, [x20, #:dtprel_lo12_nc:var]
// CHECK: ldr     w23, [x22, #:dtprel_lo12:var] // encoding: [0xd7'A',0x02'A',0x40'A',0xb9'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_lo12:var, kind: fixup_a64_ldst32_dtprel_lo12
// CHECK-NEXT: ldrsw   x21, [x20, #:dtprel_lo12_nc:var] // encoding: [0x95'A',0x02'A',0x80'A',0xb9'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_lo12_nc:var, kind: fixup_a64_ldst32_dtprel_lo12_n

// CHECK-ELF: # Relocation 26
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000068)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000217)
// CHECK-ELF: # Relocation 27
// CHECK-ELF-NEXT: (('r_offset', 0x000000000000006c)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000218)

// CHECK-ELF-NAMES: 104 R_AARCH64_TLSLD_LDST32_DTPREL_LO12
// CHECK-ELF-NAMES: 108 R_AARCH64_TLSLD_LDST32_DTPREL_LO12_NC

        ldr x19, [x18, #:dtprel_lo12:var]
        str x17, [x16, #:dtprel_lo12_nc:var]
// CHECK: ldr     x19, [x18, #:dtprel_lo12:var] // encoding: [0x53'A',0x02'A',0x40'A',0xf9'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_lo12:var, kind: fixup_a64_ldst64_dtprel_lo12
// CHECK-NEXT: str     x17, [x16, #:dtprel_lo12_nc:var] // encoding: [0x11'A',0x02'A',A,0xf9'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_lo12_nc:var, kind: fixup_a64_ldst64_dtprel_lo12_nc


// CHECK-ELF: # Relocation 28
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000070)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000219)
// CHECK-ELF: # Relocation 29
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000074)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000021a)

// CHECK-ELF-NAMES: 112 R_AARCH64_TLSLD_LDST64_DTPREL_LO12
// CHECK-ELF-NAMES: 116 R_AARCH64_TLSLD_LDST64_DTPREL_LO12_NC

        // TLS initial-exec forms
        movz x15, #:gottprel_g1:var
        movz w14, #:gottprel_g1:var
// CHECK: movz    x15, #:gottprel_g1:var  // encoding: [0x0f'A',A,0xa0'A',0x92'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :gottprel_g1:var, kind: fixup_a64_movw_gottprel_g1
// CHECK-NEXT: movz    w14, #:gottprel_g1:var  // encoding: [0x0e'A',A,0xa0'A',0x12'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :gottprel_g1:var, kind: fixup_a64_movw_gottprel_g1

// CHECK-ELF: # Relocation 30
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000078)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000021b)
// CHECK-ELF: # Relocation 31
// CHECK-ELF-NEXT: (('r_offset', 0x000000000000007c)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000021b)

// CHECK-ELF-NAMES: 120 R_AARCH64_TLSIE_MOVW_GOTTPREL_G1
// CHECK-ELF-NAMES: 124 R_AARCH64_TLSIE_MOVW_GOTTPREL_G1

        movk x13, #:gottprel_g0_nc:var
        movk w12, #:gottprel_g0_nc:var
// CHECK: movk    x13, #:gottprel_g0_nc:var // encoding: [0x0d'A',A,0x80'A',0xf2'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :gottprel_g0_nc:var, kind: fixup_a64_movw_gottprel_g0_nc
// CHECK-NEXT: movk    w12, #:gottprel_g0_nc:var // encoding: [0x0c'A',A,0x80'A',0x72'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :gottprel_g0_nc:var, kind: fixup_a64_movw_gottprel_g0_nc

// CHECK-ELF: # Relocation 32
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000080)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000021c)
// CHECK-ELF: # Relocation 33
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000084)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000021c)

// CHECK-ELF-NAMES: 128 R_AARCH64_TLSIE_MOVW_GOTTPREL_G0_NC
// CHECK-ELF-NAMES: 132 R_AARCH64_TLSIE_MOVW_GOTTPREL_G0_NC

        adrp x11, :gottprel:var
        ldr x10, [x0, #:gottprel_lo12:var]
        ldr x9, :gottprel:var
// CHECK: adrp    x11, :gottprel:var      // encoding: [0x0b'A',A,A,0x90'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :gottprel:var, kind: fixup_a64_adr_gottprel_page
// CHECK-NEXT: ldr     x10, [x0, #:gottprel_lo12:var] // encoding: [0x0a'A',A,0x40'A',0xf9'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :gottprel_lo12:var, kind: fixup_a64_ld64_gottprel_lo12_nc
// CHECK-NEXT: ldr     x9, :gottprel:var       // encoding: [0x09'A',A,A,0x58'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :gottprel:var, kind: fixup_a64_ld_gottprel_prel19

// CHECK-ELF: # Relocation 34
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000088)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000021d)
// CHECK-ELF: # Relocation 35
// CHECK-ELF-NEXT: (('r_offset', 0x000000000000008c)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000021e)
// CHECK-ELF: # Relocation 36
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000090)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000021f)

// CHECK-ELF-NAMES: 136 R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE
// CHECK-ELF-NAMES: 140 R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC
// CHECK-ELF-NAMES: 144 R_AARCH64_TLSIE_LD_GOTTPREL_PREL19

        // TLS local-exec forms
        movz x3, #:tprel_g2:var
        movn x4, #:tprel_g2:var
// CHECK: movz    x3, #:tprel_g2:var      // encoding: [0x03'A',A,0xc0'A',0x92'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_g2:var, kind: fixup_a64_movw_tprel_g2
// CHECK-NEXT: movn    x4, #:tprel_g2:var      // encoding: [0x04'A',A,0xc0'A',0x92'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_g2:var, kind: fixup_a64_movw_tprel_g2

// CHECK-ELF: # Relocation 37
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000094)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000220)
// CHECK-ELF: # Relocation 38
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000098)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000220)

// CHECK-ELF-NAMES: 148 R_AARCH64_TLSLE_MOVW_TPREL_G2
// CHECK-ELF-NAMES: 152 R_AARCH64_TLSLE_MOVW_TPREL_G2

        movz x5, #:tprel_g1:var
        movn x6, #:tprel_g1:var
        movz w7, #:tprel_g1:var
        movn w8, #:tprel_g1:var
// CHECK: movz    x5, #:tprel_g1:var      // encoding: [0x05'A',A,0xa0'A',0x92'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_g1:var, kind: fixup_a64_movw_tprel_g1
// CHECK-NEXT: movn    x6, #:tprel_g1:var      // encoding: [0x06'A',A,0xa0'A',0x92'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_g1:var, kind: fixup_a64_movw_tprel_g1
// CHECK-NEXT: movz    w7, #:tprel_g1:var      // encoding: [0x07'A',A,0xa0'A',0x12'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_g1:var, kind: fixup_a64_movw_tprel_g1
// CHECK-NEXT: movn    w8, #:tprel_g1:var      // encoding: [0x08'A',A,0xa0'A',0x12'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_g1:var, kind: fixup_a64_movw_tprel_g1

// CHECK-ELF: # Relocation 39
// CHECK-ELF-NEXT: (('r_offset', 0x000000000000009c)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000221)
// CHECK-ELF: # Relocation 40
// CHECK-ELF-NEXT: (('r_offset', 0x00000000000000a0)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000221)
// CHECK-ELF: # Relocation 41
// CHECK-ELF-NEXT: (('r_offset', 0x00000000000000a4)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000221)
// CHECK-ELF: # Relocation 42
// CHECK-ELF-NEXT: (('r_offset', 0x00000000000000a8)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000221)

// CHECK-ELF-NAMES: 156 R_AARCH64_TLSLE_MOVW_TPREL_G1
// CHECK-ELF-NAMES: 160 R_AARCH64_TLSLE_MOVW_TPREL_G1
// CHECK-ELF-NAMES: 164 R_AARCH64_TLSLE_MOVW_TPREL_G1
// CHECK-ELF-NAMES: 168 R_AARCH64_TLSLE_MOVW_TPREL_G1

        movk x9, #:tprel_g1_nc:var
        movk w10, #:tprel_g1_nc:var
// CHECK: movk    x9, #:tprel_g1_nc:var   // encoding: [0x09'A',A,0xa0'A',0xf2'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_g1_nc:var, kind: fixup_a64_movw_tprel_g1_nc
// CHECK-NEXT: movk    w10, #:tprel_g1_nc:var  // encoding: [0x0a'A',A,0xa0'A',0x72'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_g1_nc:var, kind: fixup_a64_movw_tprel_g1_nc

// CHECK-ELF: # Relocation 43
// CHECK-ELF-NEXT: (('r_offset', 0x00000000000000ac)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000222)
// CHECK-ELF: # Relocation 44
// CHECK-ELF-NEXT: (('r_offset', 0x00000000000000b0)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000222)

// CHECK-ELF-NAMES: 172 R_AARCH64_TLSLE_MOVW_TPREL_G1_NC
// CHECK-ELF-NAMES: 176 R_AARCH64_TLSLE_MOVW_TPREL_G1_NC

        movz x11, #:tprel_g0:var
        movn x12, #:tprel_g0:var
        movz w13, #:tprel_g0:var
        movn w14, #:tprel_g0:var
// CHECK: movz    x11, #:tprel_g0:var     // encoding: [0x0b'A',A,0x80'A',0x92'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_g0:var, kind: fixup_a64_movw_tprel_g0
// CHECK-NEXT: movn    x12, #:tprel_g0:var     // encoding: [0x0c'A',A,0x80'A',0x92'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_g0:var, kind: fixup_a64_movw_tprel_g0
// CHECK-NEXT: movz    w13, #:tprel_g0:var     // encoding: [0x0d'A',A,0x80'A',0x12'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_g0:var, kind: fixup_a64_movw_tprel_g0
// CHECK-NEXT: movn    w14, #:tprel_g0:var     // encoding: [0x0e'A',A,0x80'A',0x12'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_g0:var, kind: fixup_a64_movw_tprel_g0

// CHECK-ELF: # Relocation 45
// CHECK-ELF-NEXT: (('r_offset', 0x00000000000000b4)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000223)
// CHECK-ELF: # Relocation 46
// CHECK-ELF-NEXT: (('r_offset', 0x00000000000000b8)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000223)
// CHECK-ELF: # Relocation 47
// CHECK-ELF-NEXT: (('r_offset', 0x00000000000000bc)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000223)
// CHECK-ELF: # Relocation 48
// CHECK-ELF-NEXT: (('r_offset', 0x00000000000000c0)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000223)

// CHECK-ELF-NAMES: 180 R_AARCH64_TLSLE_MOVW_TPREL_G0
// CHECK-ELF-NAMES: 184 R_AARCH64_TLSLE_MOVW_TPREL_G0
// CHECK-ELF-NAMES: 188 R_AARCH64_TLSLE_MOVW_TPREL_G0
// CHECK-ELF-NAMES: 192 R_AARCH64_TLSLE_MOVW_TPREL_G0

        movk x15, #:tprel_g0_nc:var
        movk w16, #:tprel_g0_nc:var
// CHECK: movk    x15, #:tprel_g0_nc:var  // encoding: [0x0f'A',A,0x80'A',0xf2'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_g0_nc:var, kind: fixup_a64_movw_tprel_g0_nc
// CHECK-NEXT: movk    w16, #:tprel_g0_nc:var  // encoding: [0x10'A',A,0x80'A',0x72'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_g0_nc:var, kind: fixup_a64_movw_tprel_g0_nc

// CHECK-ELF: # Relocation 49
// CHECK-ELF-NEXT: (('r_offset', 0x00000000000000c4)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000224)
// CHECK-ELF: # Relocation 50
// CHECK-ELF-NEXT: (('r_offset', 0x00000000000000c8)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000224)

// CHECK-ELF-NAMES: 196 R_AARCH64_TLSLE_MOVW_TPREL_G0_NC
// CHECK-ELF-NAMES: 200 R_AARCH64_TLSLE_MOVW_TPREL_G0_NC

        add x17, x18, #:tprel_hi12:var, lsl #12
        add w19, w20, #:tprel_hi12:var, lsl #12
// CHECK: add     x17, x18, #:tprel_hi12:var, lsl #12 // encoding: [0x51'A',0x02'A',0x40'A',0x91'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_hi12:var, kind: fixup_a64_add_tprel_hi12
// CHECK-NEXT: add     w19, w20, #:tprel_hi12:var, lsl #12 // encoding: [0x93'A',0x02'A',0x40'A',0x11'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_hi12:var, kind: fixup_a64_add_tprel_hi12

// CHECK-ELF: # Relocation 51
// CHECK-ELF-NEXT: (('r_offset', 0x00000000000000cc)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000225)
// CHECK-ELF: # Relocation 52
// CHECK-ELF-NEXT: (('r_offset', 0x00000000000000d0)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000225)

// CHECK-ELF-NAMES: 204 R_AARCH64_TLSLE_ADD_TPREL_HI12
// CHECK-ELF-NAMES: 208 R_AARCH64_TLSLE_ADD_TPREL_HI12

        add x21, x22, #:tprel_lo12:var
        add w23, w24, #:tprel_lo12:var
// CHECK: add     x21, x22, #:tprel_lo12:var // encoding: [0xd5'A',0x02'A',A,0x91'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_lo12:var, kind: fixup_a64_add_tprel_lo12
// CHECK-NEXT: add     w23, w24, #:tprel_lo12:var // encoding: [0x17'A',0x03'A',A,0x11'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_lo12:var, kind: fixup_a64_add_tprel_lo12

// CHECK-ELF: # Relocation 53
// CHECK-ELF-NEXT: (('r_offset', 0x00000000000000d4)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000226)
// CHECK-ELF: # Relocation 54
// CHECK-ELF-NEXT: (('r_offset', 0x00000000000000d8)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000226)

// CHECK-ELF-NAMES: 212 R_AARCH64_TLSLE_ADD_TPREL_LO12
// CHECK-ELF-NAMES: 216 R_AARCH64_TLSLE_ADD_TPREL_LO12

        add x25, x26, #:tprel_lo12_nc:var
        add w27, w28, #:tprel_lo12_nc:var
// CHECK: add     x25, x26, #:tprel_lo12_nc:var // encoding: [0x59'A',0x03'A',A,0x91'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_lo12_nc:var, kind: fixup_a64_add_tprel_lo12_nc
// CHECK-NEXT: add     w27, w28, #:tprel_lo12_nc:var // encoding: [0x9b'A',0x03'A',A,0x11'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_lo12_nc:var, kind: fixup_a64_add_tprel_lo12_nc

// CHECK-ELF: # Relocation 55
// CHECK-ELF-NEXT: (('r_offset', 0x00000000000000dc)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000227)
// CHECK-ELF: # Relocation 56
// CHECK-ELF-NEXT: (('r_offset', 0x00000000000000e0)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000227)


// CHECK-ELF-NAMES: 220 R_AARCH64_TLSLE_ADD_TPREL_LO12_NC
// CHECK-ELF-NAMES: 224 R_AARCH64_TLSLE_ADD_TPREL_LO12_NC

        ldrb w29, [x30, #:tprel_lo12:var]
        ldrsb x29, [x28, #:tprel_lo12_nc:var]
// CHECK: ldrb    w29, [x30, #:tprel_lo12:var] // encoding: [0xdd'A',0x03'A',0x40'A',0x39'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_lo12:var, kind: fixup_a64_ldst8_tprel_lo12
// CHECK-NEXT: ldrsb   x29, [x28, #:tprel_lo12_nc:var] // encoding: [0x9d'A',0x03'A',0x80'A',0x39'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_lo12_nc:var, kind: fixup_a64_ldst8_tprel_lo12_nc

// CHECK-ELF: # Relocation 57
// CHECK-ELF-NEXT: (('r_offset', 0x00000000000000e4)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000228)
// CHECK-ELF: # Relocation 58
// CHECK-ELF-NEXT: (('r_offset', 0x00000000000000e8)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000229)

// CHECK-ELF-NAMES: 228 R_AARCH64_TLSLE_LDST8_TPREL_LO12
// CHECK-ELF-NAMES: 232 R_AARCH64_TLSLE_LDST8_TPREL_LO12_NC

        strh w27, [x26, #:tprel_lo12:var]
        ldrsh x25, [x24, #:tprel_lo12_nc:var]
// CHECK: strh    w27, [x26, #:tprel_lo12:var] // encoding: [0x5b'A',0x03'A',A,0x79'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_lo12:var, kind: fixup_a64_ldst16_tprel_lo12
// CHECK-NEXT: ldrsh   x25, [x24, #:tprel_lo12_nc:var] // encoding: [0x19'A',0x03'A',0x80'A',0x79'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_lo12_nc:var, kind: fixup_a64_ldst16_tprel_lo12_n

// CHECK-ELF: # Relocation 59
// CHECK-ELF-NEXT: (('r_offset', 0x00000000000000ec)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000022a)
// CHECK-ELF: # Relocation 60
// CHECK-ELF-NEXT: (('r_offset', 0x00000000000000f0)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000022b)

// CHECK-ELF-NAMES: 236 R_AARCH64_TLSLE_LDST16_TPREL_LO12
// CHECK-ELF-NAMES: 240 R_AARCH64_TLSLE_LDST16_TPREL_LO12_NC

        ldr w23, [x22, #:tprel_lo12:var]
        ldrsw x21, [x20, #:tprel_lo12_nc:var]
// CHECK: ldr     w23, [x22, #:tprel_lo12:var] // encoding: [0xd7'A',0x02'A',0x40'A',0xb9'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_lo12:var, kind: fixup_a64_ldst32_tprel_lo12
// CHECK-NEXT: ldrsw   x21, [x20, #:tprel_lo12_nc:var] // encoding: [0x95'A',0x02'A',0x80'A',0xb9'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_lo12_nc:var, kind: fixup_a64_ldst32_tprel_lo12_n

// CHECK-ELF: # Relocation 61
// CHECK-ELF-NEXT: (('r_offset', 0x00000000000000f4)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000022c)
// CHECK-ELF: # Relocation 62
// CHECK-ELF-NEXT: (('r_offset', 0x00000000000000f8)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000022d)

// CHECK-ELF-NAMES: 244 R_AARCH64_TLSLE_LDST32_TPREL_LO12
// CHECK-ELF-NAMES: 248 R_AARCH64_TLSLE_LDST32_TPREL_LO12_NC

        ldr x19, [x18, #:tprel_lo12:var]
        str x17, [x16, #:tprel_lo12_nc:var]
// CHECK: ldr     x19, [x18, #:tprel_lo12:var] // encoding: [0x53'A',0x02'A',0x40'A',0xf9'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_lo12:var, kind: fixup_a64_ldst64_tprel_lo12
// CHECK-NEXT: str     x17, [x16, #:tprel_lo12_nc:var] // encoding: [0x11'A',0x02'A',A,0xf9'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tprel_lo12_nc:var, kind: fixup_a64_ldst64_tprel_lo12_nc

// CHECK-ELF: # Relocation 63
// CHECK-ELF-NEXT: (('r_offset', 0x00000000000000fc)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000022e)
// CHECK-ELF: # Relocation 64
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000100)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x0000022f)

// CHECK-ELF-NAMES: 252 R_AARCH64_TLSLE_LDST64_TPREL_LO12
// CHECK-ELF-NAMES: 256 R_AARCH64_TLSLE_LDST64_TPREL_LO12_NC

        // TLS descriptor forms
        adrp x8, :tlsdesc:var
        ldr x7, [x6, :tlsdesc_lo12:var]
        add x5, x4, #:tlsdesc_lo12:var
        .tlsdesccall var
        blr x3

// CHECK: adrp    x8, :tlsdesc:var        // encoding: [0x08'A',A,A,0x90'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tlsdesc:var, kind: fixup_a64_tlsdesc_adr_page
// CHECK-NEXT: ldr     x7, [x6, #:tlsdesc_lo12:var] // encoding: [0xc7'A',A,0x40'A',0xf9'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tlsdesc_lo12:var, kind: fixup_a64_tlsdesc_ld64_lo12_nc
// CHECK-NEXT: add     x5, x4, #:tlsdesc_lo12:var // encoding: [0x85'A',A,A,0x91'A']
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tlsdesc_lo12:var, kind: fixup_a64_tlsdesc_add_lo12_nc
// CHECK-NEXT: .tlsdesccall var                // encoding: []
// CHECK-NEXT:                                 //   fixup A - offset: 0, value: :tlsdesc:var, kind: fixup_a64_tlsdesc_call
// CHECK: blr     x3                      // encoding: [0x60,0x00,0x3f,0xd6]


// CHECK-ELF: # Relocation 65
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000104)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000232)
// CHECK-ELF: # Relocation 66
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000108)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000233)
// CHECK-ELF: # Relocation 67
// CHECK-ELF-NEXT: (('r_offset', 0x000000000000010c)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000234)
// CHECK-ELF: # Relocation 68
// CHECK-ELF-NEXT: (('r_offset', 0x0000000000000110)
// CHECK-ELF-NEXT:  ('r_sym', [[VARSYM]])
// CHECK-ELF-NEXT:  ('r_type', 0x00000239)

// CHECK-ELF-NAMES: 260 R_AARCH64_TLSDESC_ADR_PAGE
// CHECK-ELF-NAMES: 264 R_AARCH64_TLSDESC_LD64_LO12_NC
// CHECK-ELF-NAMES: 268 R_AARCH64_TLSDESC_ADD_LO12_NC
// CHECK-ELF-NAMES: 272 R_AARCH64_TLSDESC_CALL


// Make sure symbol 5 has type STT_TLS:

// CHECK-ELF: # Symbol 5
// CHECK-ELF-NEXT: (('st_name', 0x00000006) # 'var'
// CHECK-ELF-NEXT:  ('st_bind', 0x1)
// CHECK-ELF-NEXT:  ('st_type', 0x6)
