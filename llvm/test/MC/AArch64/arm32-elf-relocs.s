// RUN: llvm-mc -triple=arm64-linux-gnu_ilp32 -o - < %s | FileCheck %s
// RUN: llvm-mc -triple=arm64-linux-gnu_ilp32 -show-encoding \
// RUN:    -o - < %s | FileCheck --check-prefix=CHECK-ENCODING %s
// RUN: llvm-mc -triple=arm64-linux-gnu_ilp32 -filetype=obj < %s | \
// RUN:   llvm-objdump --triple=arm64-linux-gnu - -r | \
// RUN:   FileCheck %s --check-prefix=CHECK-OBJ-ILP32

   add x0, x2, #:lo12:sym
// CHECK: add x0, x2, :lo12:sym
// CHECK-OBJ-ILP32: 0 R_AARCH64_P32_ADD_ABS_LO12_NC sym

   add x5, x7, #:dtprel_lo12:sym
// CHECK: add x5, x7, :dtprel_lo12:sym
// CHECK-OBJ-ILP32: 4 R_AARCH64_P32_TLSLD_ADD_DTPREL_LO12 sym

   add x9, x12, #:dtprel_lo12_nc:sym
// CHECK: add x9, x12, :dtprel_lo12_nc:sym
// CHECK-OBJ-ILP32: 8 R_AARCH64_P32_TLSLD_ADD_DTPREL_LO12_NC sym

   add x20, x30, #:tprel_lo12:sym
// CHECK: add x20, x30, :tprel_lo12:sym
// CHECK-OBJ-ILP32: c R_AARCH64_P32_TLSLE_ADD_TPREL_LO12 sym

   add x9, x12, #:tprel_lo12_nc:sym
// CHECK: add x9, x12, :tprel_lo12_nc:sym
// CHECK-OBJ-ILP32: 10 R_AARCH64_P32_TLSLE_ADD_TPREL_LO12_NC sym

   add x5, x0, #:tlsdesc_lo12:sym
// CHECK: add x5, x0, :tlsdesc_lo12:sym
// CHECK-OBJ-ILP32: 14 R_AARCH64_P32_TLSDESC_ADD_LO12 sym

        add x0, x2, #:lo12:sym+8
// CHECK: add x0, x2, :lo12:sym
// CHECK-OBJ-ILP32: 18 R_AARCH64_P32_ADD_ABS_LO12_NC sym+0x8

   add x5, x7, #:dtprel_lo12:sym+1
// CHECK: add x5, x7, :dtprel_lo12:sym+1
// CHECK-OBJ-ILP32: 1c R_AARCH64_P32_TLSLD_ADD_DTPREL_LO12 sym+0x1

   add x9, x12, #:dtprel_lo12_nc:sym+2
// CHECK: add x9, x12, :dtprel_lo12_nc:sym+2
// CHECK-OBJ-ILP32:20 R_AARCH64_P32_TLSLD_ADD_DTPREL_LO12_NC sym+0x2

   add x20, x30, #:tprel_lo12:sym+12
// CHECK: add x20, x30, :tprel_lo12:sym+12
// CHECK-OBJ-ILP32: 24 R_AARCH64_P32_TLSLE_ADD_TPREL_LO12 sym+0xc

   add x9, x12, #:tprel_lo12_nc:sym+54
// CHECK: add x9, x12, :tprel_lo12_nc:sym+54
// CHECK-OBJ-ILP32: 28 R_AARCH64_P32_TLSLE_ADD_TPREL_LO12_NC sym+0x36

   add x5, x0, #:tlsdesc_lo12:sym+70
// CHECK: add x5, x0, :tlsdesc_lo12:sym+70
// CHECK-OBJ-ILP32: 2c R_AARCH64_P32_TLSDESC_ADD_LO12 sym+0x46

        .hword sym + 4 - .
// CHECK-OBJ-ILP32: 30 R_AARCH64_P32_PREL16 sym+0x4
        .word sym - . + 8
// CHECK-OBJ-ILP32: 32 R_AARCH64_P32_PREL32 sym+0x8

        .hword sym
// CHECK-OBJ-ILP32: 36 R_AARCH64_P32_ABS16 sym
        .word sym+1
// CHECK-OBJ-ILP32: 38 R_AARCH64_P32_ABS32 sym+0x1

   adrp x0, sym
// CHECK: adrp x0, sym
// CHECK-OBJ-ILP32: 3c R_AARCH64_P32_ADR_PREL_PG_HI21 sym

   adrp x15, :got:sym
// CHECK: adrp x15, :got:sym
// CHECK-OBJ-ILP32: 40 R_AARCH64_P32_ADR_GOT_PAGE sym

   adrp x29, :gottprel:sym
// CHECK: adrp x29, :gottprel:sym
// CHECK-OBJ-ILP32: 44 R_AARCH64_P32_TLSIE_ADR_GOTTPREL_PAGE21 sym

   adrp x2, :tlsdesc:sym
// CHECK: adrp x2, :tlsdesc:sym
// CHECK-OBJ-ILP32: 48 R_AARCH64_P32_TLSDESC_ADR_PAGE21 sym

   // LLVM is not competent enough to do this relocation because the
   // page boundary could occur anywhere after linking. A relocation
   // is needed.
   adrp x3, trickQuestion
   .global trickQuestion
trickQuestion:
// CHECK: adrp x3, trickQuestion
// CHECK-OBJ-ILP32: 4c R_AARCH64_P32_ADR_PREL_PG_HI21 trickQuestion

   ldrb w2, [x3, :lo12:sym]
   ldrsb w5, [x7, #:lo12:sym]
   ldrsb x11, [x13, :lo12:sym]
   ldr b17, [x19, #:lo12:sym]
// CHECK: ldrb w2, [x3, :lo12:sym]
// CHECK: ldrsb w5, [x7, :lo12:sym]
// CHECK: ldrsb x11, [x13, :lo12:sym]
// CHECK: ldr b17, [x19, :lo12:sym]
// CHECK-OBJ-ILP32: R_AARCH64_P32_LDST8_ABS_LO12_NC sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_LDST8_ABS_LO12_NC sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_LDST8_ABS_LO12_NC sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_LDST8_ABS_LO12_NC sym

   ldrb w23, [x29, #:dtprel_lo12_nc:sym]
   ldrsb w23, [x19, #:dtprel_lo12:sym]
   ldrsb x17, [x13, :dtprel_lo12_nc:sym]
   ldr b11, [x7, #:dtprel_lo12:sym]
// CHECK: ldrb w23, [x29, :dtprel_lo12_nc:sym]
// CHECK: ldrsb w23, [x19, :dtprel_lo12:sym]
// CHECK: ldrsb x17, [x13, :dtprel_lo12_nc:sym]
// CHECK: ldr b11, [x7, :dtprel_lo12:sym]
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLD_LDST8_DTPREL_LO12_NC sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLD_LDST8_DTPREL_LO12 sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLD_LDST8_DTPREL_LO12_NC sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLD_LDST8_DTPREL_LO12 sym

   ldrb w1, [x2, :tprel_lo12:sym]
   ldrsb w3, [x4, #:tprel_lo12_nc:sym]
   ldrsb x5, [x6, :tprel_lo12:sym]
   ldr b7, [x8, #:tprel_lo12_nc:sym]
// CHECK: ldrb w1, [x2, :tprel_lo12:sym]
// CHECK: ldrsb w3, [x4, :tprel_lo12_nc:sym]
// CHECK: ldrsb x5, [x6, :tprel_lo12:sym]
// CHECK: ldr b7, [x8, :tprel_lo12_nc:sym]
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLE_LDST8_TPREL_LO12 sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLE_LDST8_TPREL_LO12_NC sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLE_LDST8_TPREL_LO12 sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLE_LDST8_TPREL_LO12_NC sym

   ldrh w2, [x3, #:lo12:sym]
   ldrsh w5, [x7, :lo12:sym]
   ldrsh x11, [x13, #:lo12:sym]
   ldr h17, [x19, :lo12:sym]
// CHECK: ldrh w2, [x3, :lo12:sym]
// CHECK: ldrsh w5, [x7, :lo12:sym]
// CHECK: ldrsh x11, [x13, :lo12:sym]
// CHECK: ldr h17, [x19, :lo12:sym]
// CHECK-OBJ-ILP32: R_AARCH64_P32_LDST16_ABS_LO12_NC sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_LDST16_ABS_LO12_NC sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_LDST16_ABS_LO12_NC sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_LDST16_ABS_LO12_NC sym

   ldrh w23, [x29, #:dtprel_lo12_nc:sym]
   ldrsh w23, [x19, :dtprel_lo12:sym]
   ldrsh x17, [x13, :dtprel_lo12_nc:sym]
   ldr h11, [x7, #:dtprel_lo12:sym]
// CHECK: ldrh w23, [x29, :dtprel_lo12_nc:sym]
// CHECK: ldrsh w23, [x19, :dtprel_lo12:sym]
// CHECK: ldrsh x17, [x13, :dtprel_lo12_nc:sym]
// CHECK: ldr h11, [x7, :dtprel_lo12:sym]
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLD_LDST16_DTPREL_LO12_NC sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLD_LDST16_DTPREL_LO12 sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLD_LDST16_DTPREL_LO12_NC sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLD_LDST16_DTPREL_LO12 sym

   ldrh w1, [x2, :tprel_lo12:sym]
   ldrsh w3, [x4, #:tprel_lo12_nc:sym]
   ldrsh x5, [x6, :tprel_lo12:sym]
   ldr h7, [x8, #:tprel_lo12_nc:sym]
// CHECK: ldrh w1, [x2, :tprel_lo12:sym]
// CHECK: ldrsh w3, [x4, :tprel_lo12_nc:sym]
// CHECK: ldrsh x5, [x6, :tprel_lo12:sym]
// CHECK: ldr h7, [x8, :tprel_lo12_nc:sym]
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLE_LDST16_TPREL_LO12 sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLE_LDST16_TPREL_LO12_NC sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLE_LDST16_TPREL_LO12 sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLE_LDST16_TPREL_LO12_NC sym

   ldr w1, [x2, #:lo12:sym]
   ldrsw x3, [x4, #:lo12:sym]
   ldr s4, [x5, :lo12:sym]
// CHECK: ldr w1, [x2, :lo12:sym]
// CHECK: ldrsw x3, [x4, :lo12:sym]
// CHECK: ldr s4, [x5, :lo12:sym]
// CHECK-OBJ-ILP32: R_AARCH64_P32_LDST32_ABS_LO12_NC sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_LDST32_ABS_LO12_NC sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_LDST32_ABS_LO12_NC sym

   ldr w1, [x2, :dtprel_lo12:sym]
   ldrsw x3, [x4, #:dtprel_lo12_nc:sym]
   ldr s4, [x5, #:dtprel_lo12_nc:sym]
// CHECK: ldr w1, [x2, :dtprel_lo12:sym]
// CHECK: ldrsw x3, [x4, :dtprel_lo12_nc:sym]
// CHECK: ldr s4, [x5, :dtprel_lo12_nc:sym]
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLD_LDST32_DTPREL_LO12 sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLD_LDST32_DTPREL_LO12_NC sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLD_LDST32_DTPREL_LO12_NC sym


   ldr w1, [x2, #:tprel_lo12:sym]
   ldrsw x3, [x4, :tprel_lo12_nc:sym]
   ldr s4, [x5, :tprel_lo12_nc:sym]
// CHECK: ldr w1, [x2, :tprel_lo12:sym]
// CHECK: ldrsw x3, [x4, :tprel_lo12_nc:sym]
// CHECK: ldr s4, [x5, :tprel_lo12_nc:sym]
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLE_LDST32_TPREL_LO12 sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLE_LDST32_TPREL_LO12_NC sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLE_LDST32_TPREL_LO12_NC sym

   ldr x28, [x27, :lo12:sym]
   ldr d26, [x25, :lo12:sym]
// CHECK: ldr x28, [x27, :lo12:sym]
// CHECK: ldr d26, [x25, :lo12:sym]
// CHECK-OBJ-ILP32: R_AARCH64_P32_LDST64_ABS_LO12_NC sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_LDST64_ABS_LO12_NC sym

   ldr w24, [x23, :got_lo12:sym]
   ldr s22, [x21, :got_lo12:sym]
// CHECK: ldr w24, [x23, :got_lo12:sym]
// CHECK: ldr s22, [x21, :got_lo12:sym]
// CHECK-OBJ-ILP32: R_AARCH64_P32_LD32_GOT_LO12_NC sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_LD32_GOT_LO12_NC sym

   ldr x24, [x23, :dtprel_lo12_nc:sym]
   ldr d22, [x21, :dtprel_lo12:sym]
// CHECK: ldr x24, [x23, :dtprel_lo12_nc:sym]
// CHECK: ldr d22, [x21, :dtprel_lo12:sym]
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLD_LDST64_DTPREL_LO12_NC sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLD_LDST64_DTPREL_LO12 sym

   ldr q24, [x23, :dtprel_lo12_nc:sym]
   ldr q22, [x21, :dtprel_lo12:sym]
// CHECK: ldr q24, [x23, :dtprel_lo12_nc:sym]
// CHECK: ldr q22, [x21, :dtprel_lo12:sym]
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLD_LDST128_DTPREL_LO12_NC sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLD_LDST128_DTPREL_LO12 sym

   ldr x24, [x23, :tprel_lo12:sym]
   ldr d22, [x21, :tprel_lo12_nc:sym]
// CHECK: ldr x24, [x23, :tprel_lo12:sym]
// CHECK: ldr d22, [x21, :tprel_lo12_nc:sym]
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLE_LDST64_TPREL_LO12 sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLE_LDST64_TPREL_LO12_NC sym

   ldr q24, [x23, :tprel_lo12:sym]
   ldr q22, [x21, :tprel_lo12_nc:sym]
// CHECK: ldr q24, [x23, :tprel_lo12:sym]
// CHECK: ldr q22, [x21, :tprel_lo12_nc:sym]
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLE_LDST128_TPREL_LO12 sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSLE_LDST128_TPREL_LO12_NC sym

   ldr w24, [x23, :gottprel_lo12:sym]
   ldr s22, [x21, :gottprel_lo12:sym]

   ldr w24, [x23, :tlsdesc_lo12:sym]
   ldr s22, [x21, :tlsdesc_lo12:sym]
// CHECK: ldr w24, [x23, :tlsdesc_lo12:sym]
// CHECK: ldr s22, [x21, :tlsdesc_lo12:sym]
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSDESC_LD32_LO12 sym
// CHECK-OBJ-ILP32: R_AARCH64_P32_TLSDESC_LD32_LO12 sym

   ldr q20, [x19, #:lo12:sym]
// CHECK: ldr q20, [x19, :lo12:sym]
// CHECK-OBJ-ILP32: R_AARCH64_P32_LDST128_ABS_LO12_NC sym
// check encoding here, since encoding test doesn't belong with TLS encoding
// tests, as it isn't a TLS relocation.
// CHECK-ENCODING: ldr q20, [x19, :lo12:sym] // encoding: [0x74,0bAAAAAA10,0b11AAAAAA,0x3d]
// CHECK-ENCODING-NEXT:  0, value: :lo12:sym, kind: fixup_aarch64_ldst_imm12_scale16

// Since relocated instructions print without a '#', that syntax should
// certainly be accepted when assembling.
   add x3, x5, :lo12:imm
// CHECK: add x3, x5, :lo12:imm
