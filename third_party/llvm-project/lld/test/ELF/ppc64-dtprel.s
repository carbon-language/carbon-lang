// REQUIRES: ppc

// RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %s -o %t.o
// RUN: ld.lld -shared %t.o -o %t.so
// RUN: llvm-readelf -r %t.o | FileCheck --check-prefix=InputRelocs %s
// RUN: llvm-readelf -r %t.so | FileCheck --check-prefix=OutputRelocs %s
// RUN: llvm-readelf -x .got %t.so | FileCheck --check-prefix=HEX-LE %s
// RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck --check-prefix=Dis %s

// RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %s -o %t.o
// RUN: ld.lld -shared %t.o -o %t.so
// RUN: llvm-readelf -r %t.o | FileCheck --check-prefix=InputRelocs %s
// RUN: llvm-readelf -r %t.so | FileCheck --check-prefix=OutputRelocs %s
// RUN: llvm-readelf -x .got %t.so | FileCheck --check-prefix=HEX-BE %s
// RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck --check-prefix=Dis %s

        .text
        .abiversion 2
        .globl  test
        .p2align        4
        .type   test,@function
test:
.Lfunc_gep0:
        addis 2, 12, .TOC.-.Lfunc_gep0@ha
        addi 2, 2, .TOC.-.Lfunc_gep0@l
.Lfunc_lep0:
        .localentry     test, .Lfunc_lep0-.Lfunc_gep0
        mflr 0
        std 0, 16(1)
        stdu 1, -32(1)
        addis 3, 2, i@got@tlsld@ha
        addi 3, 3, i@got@tlsld@l
        bl __tls_get_addr(i@tlsld)
        nop
        addi 4, 3, i@dtprel
        lwa 4, i@dtprel(3)
        ld 0, 16(1)
        mtlr 0
        blr

        .globl test_64
        .p2align        4
        .type    test_64,@function

        .globl test_adjusted
        .p2align        4
        .type    test_adjusted,@function
test_adjusted:
.Lfunc_gep1:
        addis 2, 12, .TOC.-.Lfunc_gep1@ha
        addi 2, 2, .TOC.-.Lfunc_gep1@l
.Lfunc_lep1:
        .localentry     test_adjusted, .Lfunc_lep1-.Lfunc_gep1
        mflr 0
        std 0, 16(1)
        stdu 1, -32(1)
        addis 3, 2, k@got@tlsld@ha
        addi 3, 3, k@got@tlsld@l
        bl __tls_get_addr(k@tlsld)
        nop
        lis 4, k@dtprel@highesta
        ori 4, 4, k@dtprel@highera
        lis 5, k@dtprel@ha
        addi 5, 5, k@dtprel@l
        sldi 4, 4, 32
        or   4, 4, 5
        add  3, 3, 4
        addi 1, 1, 32
        ld 0, 16(1)
        mtlr 0
        blr

        .globl test_not_adjusted
        .p2align      4
        .type test_not_adjusted,@function
test_not_adjusted:
.Lfunc_gep2:
        addis 2, 12, .TOC.-.Lfunc_gep2@ha
        addi 2, 2, .TOC.-.Lfunc_gep2@l
.Lfunc_lep2:
        .localentry     test_not_adjusted, .Lfunc_lep2-.Lfunc_gep2
        mflr 0
        std 0, 16(1)
        stdu 1, -32(1)
        addis 3, 2, i@got@tlsld@ha
        addi 3, 3, i@got@tlsld@l
        bl __tls_get_addr(k@tlsld)
        nop
        lis 4, k@dtprel@highest
        ori 4, 4, k@dtprel@higher
        sldi 4, 4, 32
        oris  4, 4, k@dtprel@h
        ori   4, 4, k@dtprel@l
        add 3, 3, 4
        addi 1, 1, 32
        ld 0, 16(1)
        mtlr 0
        blr

        .section        .debug_addr,"",@progbits
        .quad   i@dtprel+32768

        .type   i,@object
        .section        .tdata,"awT",@progbits
        .space 1024
        .p2align        2
i:
        .long   55
        .size   i, 4

        .space 1024 * 1024 * 4
        .type k,@object
        .p2align 2
k:
       .long 128
       .size k,4

// Verify the input has all the remaining DTPREL based relocations we want to
// test.
// InputRelocs: Relocation section '.rela.text'
// InputRelocs: R_PPC64_DTPREL16          {{[0-9a-f]+}} i + 0
// InputRelocs: R_PPC64_DTPREL16_DS       {{[0-9a-f]+}} i + 0
// InputRelocs: R_PPC64_DTPREL16_HIGHESTA {{[0-9a-f]+}} k + 0
// InputRelocs: R_PPC64_DTPREL16_HIGHERA  {{[0-9a-f]+}} k + 0
// InputRelocs: R_PPC64_DTPREL16_HA       {{[0-9a-f]+}} k + 0
// InputRelocs: R_PPC64_DTPREL16_LO       {{[0-9a-f]+}} k + 0
// InputRelocs: R_PPC64_DTPREL16_HIGHEST  {{[0-9a-f]+}} k + 0
// InputRelocs: R_PPC64_DTPREL16_HIGHER   {{[0-9a-f]+}} k + 0
// InputRelocs: R_PPC64_DTPREL16_HI       {{[0-9a-f]+}} k + 0
// InputRelocs: R_PPC64_DTPREL16_LO       {{[0-9a-f]+}} k + 0
// InputRelocs: Relocation section '.rela.debug_addr'
// InputRelocs: R_PPC64_DTPREL64          {{[0-9a-f]+}} i + 8000

// Expect a single dynamic relocation in the '.rela.dyn section for the module id.
// OutputRelocs:      Relocation section '.rela.dyn' at offset 0x{{[0-9a-f]+}} contains 1 entries:
// OutputRelocs-NEXT: Offset Info Type Symbol's Value Symbol's Name + Addend
// OutputRelocs-NEXT: R_PPC64_DTPMOD64


// The got entry for i is at .got+8*1 = 0x4209e0
// i@dtprel = 1024 - 0x8000 = -31744 = 0xffffffffffff8400
// HEX-LE:      section '.got':
// HEX-LE-NEXT: 4209d0 d0894200 00000000 00000000 00000000
// HEX-LE-NEXT: 4209e0 00000000 00000000

// HEX-BE:      section '.got':
// HEX-BE-NEXT: 4209d0 00000000 004289d0 00000000 00000000
// HEX-BE-NEXT: 4209e0 00000000 00000000

// Dis:     <test>:
// Dis:      addi 4, 3, -31744
// Dis-NEXT: lwa 4, -31744(3)

// #k@dtprel(1024 + 4 + 1024 * 1024 * 4) = 0x400404

// #highesta(k@dtprel) --> ((0x400404 - 0x8000 + 0x8000) >> 48) & 0xffff = 0
// #highera(k@dtprel)  --> ((0x400404 - 0x8000 + 0x8000) >> 32) & 0xffff = 0
// #ha(k@dtprel)       --> ((0x400404 - 0x8000 + 0x8000) >> 16) & 0xffff = 64
// #lo(k@dtprel)       --> ((0x400404 - 0x8000) & 0xffff = -31740
// Dis:  <test_adjusted>:
// Dis:     lis 4, 0
// Dis:     ori 4, 4, 0
// Dis:     lis 5, 64
// Dis:     addi 5, 5, -31740

// #highest(k@dtprel) --> ((0x400404 - 0x8000) >> 48) & 0xffff = 0
// #higher(k@dtprel)  --> ((0x400404 - 0x8000) >> 32) & 0xffff = 0
// #hi(k@dtprel)      --> ((0x400404 - 0x8000) >> 16) & 0xffff = 63
// #lo(k@dtprel)      --> ((0x400404 - 0x8000) & 0xffff = 33796
// Dis:  <test_not_adjusted>:
// Dis:    lis 4, 0
// Dis:    ori 4, 4, 0
// Dis:    oris 4, 4, 63
// Dis:    ori 4, 4, 33796

