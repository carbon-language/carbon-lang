! RUN: llvm-mc %s -arch=sparcv9 --relocation-model=pic -filetype=obj | llvm-readobj -r | FileCheck %s


! CHECK:      Relocations [
! CHECK-NOT:    0x{{[0-9,A-F]+}} R_SPARC_WPLT30 .text 0xC
! CHECK:        0x{{[0-9,A-F]+}} R_SPARC_PC22 _GLOBAL_OFFSET_TABLE_ 0x4
! CHECK-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_PC10 _GLOBAL_OFFSET_TABLE_ 0x8
! CHECK-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_GOT22 AGlobalVar 0x0
! CHECK-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_GOT10 AGlobalVar 0x0
! CHECK-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_GOT22 .LC0 0x0
! CHECK-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_GOT10 .LC0 0x0
! CHECK-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_WPLT30 bar 0x0
! CHECK:      ]

        .section        ".rodata"
        .align 8
.LC0:
        .asciz   "string"
        .section ".text"
        .text
        .globl  foo
        .align  4
        .type   foo,@function
foo:
        .cfi_startproc
        save %sp, -176, %sp
        .cfi_def_cfa_register %fp
        .cfi_window_save
        .cfi_register 15, 31
.Ltmp4:
        call .Ltmp5
.Ltmp6:
        sethi %hi(_GLOBAL_OFFSET_TABLE_+(.Ltmp6-.Ltmp4)), %i1
.Ltmp5:
        or %i1, %lo(_GLOBAL_OFFSET_TABLE_+(.Ltmp5-.Ltmp4)), %i1
        add %i1, %o7, %i1
        sethi %hi(AGlobalVar), %i2
        add %i2, %lo(AGlobalVar), %i2
        ldx [%i1+%i2], %i3
        ldx [%i3], %i3
        sethi %hi(.LC0), %i2
        add %i2, %lo(.LC0), %i2
        ldx [%i1+%i2], %i4
        call bar
        add %i0, %i1, %o0
        ret
        restore %g0, %o0, %o0
.Ltmp7:
        .size   foo, .Ltmp7-foo
        .cfi_endproc

        .type   AGlobalVar,@object      ! @AGlobalVar
        .section        .bss,#alloc,#write
        .globl  AGlobalVar
        .align  8
AGlobalVar:
        .xword  0                       ! 0x0
        .size   AGlobalVar, 8
