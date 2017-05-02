// RUN: llvm-mc -target-abi=ilp32 -triple=arm64-none-linux-gnu \
// RUN:   -show-encoding < %s | FileCheck --check-prefix=CHECK-ILP32 %s
// RUN: llvm-mc -target-abi=ilp32 -triple=arm64-none-linux-gnu \
// RUN:   -filetype=obj < %s -o - | \
// RUN:   llvm-readobj -r -t | FileCheck --check-prefix=CHECK-ELF-ILP32 %s

////////////////////////////////////////////////////////////////////////////////
// TLS initial-exec forms
////////////////////////////////////////////////////////////////////////////////

        adrp x11, :gottprel:var
        ldr w10, [x0, #:gottprel_lo12:var]
        ldr w9, :gottprel:var
// CHECK-ILP32: adrp x11, :gottprel:var      // encoding: [0x0b'A',A,A,0x90'A']
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :gottprel:var, kind: fixup_aarch64_pcrel_adrp_imm21
// CHECK-ILP32: ldr  w10, [x0, :gottprel_lo12:var] // encoding: [0x0a,0bAAAAAA00,0b01AAAAAA,0xb9]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :gottprel_lo12:var, kind: fixup_aarch64_ldst_imm12_scale4
// CHECK-ILP32: ldr     w9, :gottprel:var       // encoding: [0bAAA01001,A,A,0x18]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :gottprel:var, kind: fixup_aarch64_ldr_pcrel_imm19

// CHECK-ELF-ILP32:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSIE_ADR_GOTTPREL_PAGE21 [[VARSYM:[^ ]+]]
// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSIE_LD32_GOTTPREL_LO12_NC [[VARSYM]]
// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSIE_LD_GOTTPREL_PREL19 [[VARSYM]]


////////////////////////////////////////////////////////////////////////////////
// TLS local-exec forms
////////////////////////////////////////////////////////////////////////////////

        movz x5, #:tprel_g1:var
        movn x6, #:tprel_g1:var
        movz w7, #:tprel_g1:var
// CHECK-ILP32: movz    x5, #:tprel_g1:var      // encoding: [0bAAA00101,A,0b101AAAAA,0x92]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :tprel_g1:var, kind: fixup_aarch64_movw
// CHECK-ILP32: movn    x6, #:tprel_g1:var      // encoding: [0bAAA00110,A,0b101AAAAA,0x92]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :tprel_g1:var, kind: fixup_aarch64_movw
// CHECK-ILP32: movz    w7, #:tprel_g1:var      // encoding: [0bAAA00111,A,0b101AAAAA,0x12]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :tprel_g1:var, kind: fixup_aarch64_movw

// CHECK-ELF-ILP32: {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLE_MOVW_TPREL_G1 [[VARSYM]]
// CHECK-ELF-ILP32: {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLE_MOVW_TPREL_G1 [[VARSYM]]
// CHECK-ELF-ILP32: {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLE_MOVW_TPREL_G1 [[VARSYM]]


        movz x11, #:tprel_g0:var
        movn x12, #:tprel_g0:var
        movz w13, #:tprel_g0:var
// CHECK-ILP32: movz    x11, #:tprel_g0:var     // encoding: [0bAAA01011,A,0b100AAAAA,0x92]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :tprel_g0:var, kind: fixup_aarch64_movw
// CHECK-ILP32: movn    x12, #:tprel_g0:var     // encoding: [0bAAA01100,A,0b100AAAAA,0x92]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :tprel_g0:var, kind: fixup_aarch64_movw
// CHECK-ILP32: movz    w13, #:tprel_g0:var     // encoding: [0bAAA01101,A,0b100AAAAA,0x12]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :tprel_g0:var, kind: fixup_aarch64_movw

// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLE_MOVW_TPREL_G0 [[VARSYM]]
// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLE_MOVW_TPREL_G0 [[VARSYM]]
// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLE_MOVW_TPREL_G0 [[VARSYM]]


        movk w15, #:tprel_g0_nc:var
        movk w16, #:tprel_g0_nc:var
// CHECK-ILP32: movk    w15, #:tprel_g0_nc:var  // encoding: [0bAAA01111,A,0b100AAAAA,0x72]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :tprel_g0_nc:var, kind: fixup_aarch64_movw
// CHECK-ILP32: movk    w16, #:tprel_g0_nc:var  // encoding: [0bAAA10000,A,0b100AAAAA,0x72]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :tprel_g0_nc:var, kind: fixup_aarch64_movw

// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLE_MOVW_TPREL_G0_NC [[VARSYM]]
// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLE_MOVW_TPREL_G0_NC [[VARSYM]]


        add x21, x22, #:tprel_lo12:var
// CHECK-ILP32: add     x21, x22, :tprel_lo12:var // encoding: [0xd5,0bAAAAAA10,0b00AAAAAA,0x91]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :tprel_lo12:var, kind: fixup_aarch64_add_imm12

// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLE_ADD_TPREL_LO12 [[VARSYM]]


        add x25, x26, #:tprel_lo12_nc:var
// CHECK-ILP32: add     x25, x26, :tprel_lo12_nc:var // encoding: [0x59,0bAAAAAA11,0b00AAAAAA,0x91]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :tprel_lo12_nc:var, kind: fixup_aarch64_add_imm12

// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLE_ADD_TPREL_LO12_NC [[VARSYM]]


        ldrb w29, [x30, #:tprel_lo12:var]
        ldrsb x29, [x28, #:tprel_lo12_nc:var]
// CHECK-ILP32: ldrb    w29, [x30, :tprel_lo12:var] // encoding: [0xdd,0bAAAAAA11,0b01AAAAAA,0x39]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :tprel_lo12:var, kind: fixup_aarch64_ldst_imm12_scale1
// CHECK-ILP32: ldrsb   x29, [x28, :tprel_lo12_nc:var] // encoding: [0x9d,0bAAAAAA11,0b10AAAAAA,0x39]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :tprel_lo12_nc:var, kind: fixup_aarch64_ldst_imm12_scale1

// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLE_LDST8_TPREL_LO12 [[VARSYM]]
// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLE_LDST8_TPREL_LO12_NC [[VARSYM]]


        strh w27, [x26, #:tprel_lo12:var]
        ldrsh x25, [x24, #:tprel_lo12_nc:var]
// CHECK-ILP32: strh    w27, [x26, :tprel_lo12:var] // encoding: [0x5b,0bAAAAAA11,0b00AAAAAA,0x79]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :tprel_lo12:var, kind: fixup_aarch64_ldst_imm12_scale2
// CHECK-ILP32: ldrsh   x25, [x24, :tprel_lo12_nc:var] // encoding: [0x19,0bAAAAAA11,0b10AAAAAA,0x79]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :tprel_lo12_nc:var, kind: fixup_aarch64_ldst_imm12_scale2

// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLE_LDST16_TPREL_LO12 [[VARSYM]]
// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLE_LDST16_TPREL_LO12_NC [[VARSYM]]


        ldr w23, [x22, #:tprel_lo12:var]
        ldrsw x21, [x20, #:tprel_lo12_nc:var]
// CHECK-ILP32: ldr     w23, [x22, :tprel_lo12:var] // encoding: [0xd7,0bAAAAAA10,0b01AAAAAA,0xb9]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :tprel_lo12:var, kind: fixup_aarch64_ldst_imm12_scale4
// CHECK-ILP32: ldrsw   x21, [x20, :tprel_lo12_nc:var] // encoding: [0x95,0bAAAAAA10,0b10AAAAAA,0xb9]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :tprel_lo12_nc:var, kind: fixup_aarch64_ldst_imm12_scale4

// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLE_LDST32_TPREL_LO12 [[VARSYM]]
// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLE_LDST32_TPREL_LO12_NC [[VARSYM]]

        ldr x19, [x18, #:tprel_lo12:var]
        str x17, [x16, #:tprel_lo12_nc:var]
// CHECK-ILP32: ldr     x19, [x18, :tprel_lo12:var] // encoding: [0x53,0bAAAAAA10,0b01AAAAAA,0xf9]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :tprel_lo12:var, kind: fixup_aarch64_ldst_imm12_scale8
// CHECK-ILP32: str     x17, [x16, :tprel_lo12_nc:var] // encoding: [0x11,0bAAAAAA10,0b00AAAAAA,0xf9]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :tprel_lo12_nc:var, kind: fixup_aarch64_ldst_imm12_scale8

// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLE_LDST64_TPREL_LO12 [[VARSYM]]
// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLE_LDST64_TPREL_LO12_NC [[VARSYM]]


   ldr q24, [x23, :tprel_lo12:var]
   str q22, [x21, :tprel_lo12_nc:var]
// CHECK-ILP32: ldr     q24, [x23, :tprel_lo12:var] // encoding: [0xf8,0bAAAAAA10,0b11AAAAAA,0x3d]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :tprel_lo12:var, kind: fixup_aarch64_ldst_imm12_scale16
// CHECK-ILP32: str     q22, [x21, :tprel_lo12_nc:var] // encoding: [0xb6,0bAAAAAA10,0b10AAAAAA,0x3d]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :tprel_lo12_nc:var, kind: fixup_aarch64_ldst_imm12_scale16

// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLE_LDST128_TPREL_LO12 [[VARSYM]]
// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLE_LDST128_TPREL_LO12_NC [[VARSYM]]

////////////////////////////////////////////////////////////////////////////////
// TLS local-dynamic forms
////////////////////////////////////////////////////////////////////////////////

        movz x5, #:dtprel_g1:var
        movn x6, #:dtprel_g1:var
        movz w7, #:dtprel_g1:var
// CHECK-ILP32: movz    x5, #:dtprel_g1:var      // encoding: [0bAAA00101,A,0b101AAAAA,0x92]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_g1:var, kind: fixup_aarch64_movw
// CHECK-ILP32: movn    x6, #:dtprel_g1:var      // encoding: [0bAAA00110,A,0b101AAAAA,0x92]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_g1:var, kind: fixup_aarch64_movw
// CHECK-ILP32: movz    w7, #:dtprel_g1:var      // encoding: [0bAAA00111,A,0b101AAAAA,0x12]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_g1:var, kind: fixup_aarch64_movw

// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLD_MOVW_DTPREL_G1 [[VARSYM]]
// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLD_MOVW_DTPREL_G1 [[VARSYM]]
// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLD_MOVW_DTPREL_G1 [[VARSYM]]


        movz x11, #:dtprel_g0:var
        movn x12, #:dtprel_g0:var
        movz w13, #:dtprel_g0:var
// CHECK-ILP32: movz    x11, #:dtprel_g0:var     // encoding: [0bAAA01011,A,0b100AAAAA,0x92]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_g0:var, kind: fixup_aarch64_movw
// CHECK-ILP32: movn    x12, #:dtprel_g0:var     // encoding: [0bAAA01100,A,0b100AAAAA,0x92]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_g0:var, kind: fixup_aarch64_movw
// CHECK-ILP32: movz    w13, #:dtprel_g0:var     // encoding: [0bAAA01101,A,0b100AAAAA,0x12]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_g0:var, kind: fixup_aarch64_movw

// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLD_MOVW_DTPREL_G0 [[VARSYM]]
// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLD_MOVW_DTPREL_G0 [[VARSYM]]
// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLD_MOVW_DTPREL_G0 [[VARSYM]]


        movk x15, #:dtprel_g0_nc:var
        movk w16, #:dtprel_g0_nc:var
// CHECK-ILP32: movk    x15, #:dtprel_g0_nc:var  // encoding: [0bAAA01111,A,0b100AAAAA,0xf2]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_g0_nc:var, kind: fixup_aarch64_movw
// CHECK-ILP32: movk    w16, #:dtprel_g0_nc:var  // encoding: [0bAAA10000,A,0b100AAAAA,0x72]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_g0_nc:var, kind: fixup_aarch64_movw

// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLD_MOVW_DTPREL_G0_NC [[VARSYM]]
// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLD_MOVW_DTPREL_G0_NC [[VARSYM]]


        add x21, x22, #:dtprel_lo12:var
// CHECK-ILP32: add     x21, x22, :dtprel_lo12:var // encoding: [0xd5,0bAAAAAA10,0b00AAAAAA,0x91]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_lo12:var, kind: fixup_aarch64_add_imm12

// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLD_ADD_DTPREL_LO12 [[VARSYM]]


        add x25, x26, #:dtprel_lo12_nc:var
// CHECK-ILP32: add     x25, x26, :dtprel_lo12_nc:var // encoding: [0x59,0bAAAAAA11,0b00AAAAAA,0x91]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_lo12_nc:var, kind: fixup_aarch64_add_imm12

// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLD_ADD_DTPREL_LO12_NC [[VARSYM]]


	add x0, x0, #:dtprel_hi12:var_tlsld, lsl #12
	add x0, x0, #:tprel_hi12:var_tlsle, lsl #12

// CHECK-ELF-ILP32: R_AARCH64_P32_TLSLD_ADD_DTPREL_HI12 var_tlsld
// CHECK-ELF-ILP32: R_AARCH64_P32_TLSLE_ADD_TPREL_HI12 var_tlsle


        ldrb w29, [x30, #:dtprel_lo12:var]
        ldrsb x29, [x28, #:dtprel_lo12_nc:var]
// CHECK-ILP32: ldrb    w29, [x30, :dtprel_lo12:var] // encoding: [0xdd,0bAAAAAA11,0b01AAAAAA,0x39]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_lo12:var, kind: fixup_aarch64_ldst_imm12_scale1
// CHECK-ILP32: ldrsb   x29, [x28, :dtprel_lo12_nc:var] // encoding: [0x9d,0bAAAAAA11,0b10AAAAAA,0x39]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_lo12_nc:var, kind: fixup_aarch64_ldst_imm12_scale1

// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLD_LDST8_DTPREL_LO12 [[VARSYM]]
// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLD_LDST8_DTPREL_LO12_NC [[VARSYM]]


        strh w27, [x26, #:dtprel_lo12:var]
        ldrsh x25, [x24, #:dtprel_lo12_nc:var]
// CHECK-ILP32: strh    w27, [x26, :dtprel_lo12:var] // encoding: [0x5b,0bAAAAAA11,0b00AAAAAA,0x79]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_lo12:var, kind: fixup_aarch64_ldst_imm12_scale2
// CHECK-ILP32: ldrsh   x25, [x24, :dtprel_lo12_nc:var] // encoding: [0x19,0bAAAAAA11,0b10AAAAAA,0x79]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_lo12_nc:var, kind: fixup_aarch64_ldst_imm12_scale2

// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLD_LDST16_DTPREL_LO12 [[VARSYM]]
// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLD_LDST16_DTPREL_LO12_NC [[VARSYM]]


        ldr w23, [x22, #:dtprel_lo12:var]
        ldrsw x21, [x20, #:dtprel_lo12_nc:var]
// CHECK-ILP32: ldr     w23, [x22, :dtprel_lo12:var] // encoding: [0xd7,0bAAAAAA10,0b01AAAAAA,0xb9]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_lo12:var, kind: fixup_aarch64_ldst_imm12_scale4
// CHECK-ILP32: ldrsw   x21, [x20, :dtprel_lo12_nc:var] // encoding: [0x95,0bAAAAAA10,0b10AAAAAA,0xb9]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_lo12_nc:var, kind: fixup_aarch64_ldst_imm12_scale4

// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLD_LDST32_DTPREL_LO12 [[VARSYM]]
// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLD_LDST32_DTPREL_LO12_NC [[VARSYM]]

        ldr x19, [x18, #:dtprel_lo12:var]
        str x17, [x16, #:dtprel_lo12_nc:var]
// CHECK-ILP32: ldr     x19, [x18, :dtprel_lo12:var] // encoding: [0x53,0bAAAAAA10,0b01AAAAAA,0xf9]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_lo12:var, kind: fixup_aarch64_ldst_imm12_scale8
// CHECK-ILP32: str     x17, [x16, :dtprel_lo12_nc:var] // encoding: [0x11,0bAAAAAA10,0b00AAAAAA,0xf9]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_lo12_nc:var, kind: fixup_aarch64_ldst_imm12_scale8

// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLD_LDST64_DTPREL_LO12 [[VARSYM]]
// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLD_LDST64_DTPREL_LO12_NC [[VARSYM]]

        ldr q24, [x23, #:dtprel_lo12:var]
        str q22, [x21, #:dtprel_lo12_nc:var]
// CHECK-ILP32: ldr     q24, [x23, :dtprel_lo12:var] // encoding: [0xf8,0bAAAAAA10,0b11AAAAAA,0x3d]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_lo12:var, kind: fixup_aarch64_ldst_imm12_scale16
// CHECK-ILP32: str     q22, [x21, :dtprel_lo12_nc:var] // encoding: [0xb6,0bAAAAAA10,0b10AAAAAA,0x3d]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :dtprel_lo12_nc:var, kind: fixup_aarch64_ldst_imm12_scale16

// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLD_LDST128_DTPREL_LO12 [[VARSYM]]
// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSLD_LDST128_DTPREL_LO12_NC [[VARSYM]]

////////////////////////////////////////////////////////////////////////////////
// TLS descriptor forms
////////////////////////////////////////////////////////////////////////////////

        adrp x8, :tlsdesc:var
        ldr w7, [x6, #:tlsdesc_lo12:var]
        add x5, x4, #:tlsdesc_lo12:var
        .tlsdesccall var
        blr x3

// CHECK-ILP32: adrp    x8, :tlsdesc:var        // encoding: [0x08'A',A,A,0x90'A']
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :tlsdesc:var, kind: fixup_aarch64_pcrel_adrp_imm21
// CHECK-ILP32: ldr     w7, [x6, :tlsdesc_lo12:var] // encoding: [0xc7,0bAAAAAA00,0b01AAAAAA,0xb9]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :tlsdesc_lo12:var, kind: fixup_aarch64_ldst_imm12_scale4
// CHECK-ILP32: add     x5, x4, :tlsdesc_lo12:var // encoding: [0x85,0bAAAAAA00,0b00AAAAAA,0x91]
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: :tlsdesc_lo12:var, kind: fixup_aarch64_add_imm12
// CHECK-ILP32: .tlsdesccall var                // encoding: []
// CHECK-ILP32-NEXT:                                 //   fixup A - offset: 0, value: var, kind: fixup_aarch64_tlsdesc_call
// CHECK-ILP32: blr     x3                      // encoding: [0x60,0x00,0x3f,0xd6]


// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSDESC_ADR_PAGE21 [[VARSYM]]
// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSDESC_LD32_LO12 [[VARSYM]]
// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSDESC_ADD_LO12 [[VARSYM]]
// CHECK-ELF-ILP32-NEXT:     {{0x[0-9A-F]+}} R_AARCH64_P32_TLSDESC_CALL [[VARSYM]]

        // Make sure symbol 5 has type STT_TLS:

// CHECK-ELF-ILP32:      Symbols [
// CHECK-ELF-ILP32:        Symbol {
// CHECK-ELF-ILP32:          Name: var
// CHECK-ELF-ILP32-NEXT:     Value:
// CHECK-ELF-ILP32-NEXT:     Size:
// CHECK-ELF-ILP32-NEXT:     Binding: Global
// CHECK-ELF-ILP32-NEXT:     Type: TLS
