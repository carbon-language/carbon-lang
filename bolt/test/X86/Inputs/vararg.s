        .file   "vararg.cpp"
        .section        .debug_abbrev,"",@progbits
.Ldebug_abbrev0:
        .section        .debug_info,"",@progbits
.Ldebug_info0:
        .section        .debug_line,"",@progbits
.Ldebug_line0:
        .text
.Ltext0:
.globl _Z8myprintfPKcz
        .type   _Z8myprintfPKcz, @function
_Z8myprintfPKcz:
.LFB0:
        .file 1 "./example.cpp"
        .loc 1 3 0
        .cfi_startproc
        pushq   %rbp
.LCFI0:
        .cfi_def_cfa_offset 16
        movq    %rsp, %rbp
        .cfi_offset 6, -16
.LCFI1:
        .cfi_def_cfa_register 6
        subq    $104, %rsp
        movq    %rsi, -168(%rbp)
        movq    %rdx, -160(%rbp)
        movq    %rcx, -152(%rbp)
        movq    %r8, -144(%rbp)
        movq    %r9, -136(%rbp)
        movzbl  %al, %eax
        leaq    0(,%rax,4), %rdx
        movl    $.L2, %eax
        subq    %rdx, %rax
        leaq    -1(%rbp), %rdx
        jmp     *%rax
        movaps  %xmm7, -15(%rdx)
        movaps  %xmm6, -31(%rdx)
        movaps  %xmm5, -47(%rdx)
        movaps  %xmm4, -63(%rdx)
        movaps  %xmm3, -79(%rdx)
        movaps  %xmm2, -95(%rdx)
        movaps  %xmm1, -111(%rdx)
        movaps  %xmm0, -127(%rdx)
.L2:
        movq    %rdi, -216(%rbp)
.LBB2:
        .loc 1 5 0
        leaq    -208(%rbp), %rax
        movl    $8, (%rax)
        leaq    -208(%rbp), %rax
        movl    $48, 4(%rax)
        leaq    -208(%rbp), %rax
        leaq    16(%rbp), %rdx
        movq    %rdx, 8(%rax)
        leaq    -208(%rbp), %rax
        leaq    -176(%rbp), %rdx
        movq    %rdx, 16(%rax)
        .loc 1 7 0
        movl    $1, %eax
.LBE2:
        .loc 1 8 0
        leave
.LCFI2:
        .cfi_def_cfa 7, 8
        ret
        .cfi_endproc
.LFE0:
        .size   _Z8myprintfPKcz, .-_Z8myprintfPKcz
.globl main
        .type   main, @function
main:
.LFB1:
        .loc 1 10 0
        .cfi_startproc
        pushq   %rbp
.LCFI3:
        .cfi_def_cfa_offset 16
        movq    %rsp, %rbp
        .cfi_offset 6, -16
.LCFI4:
        .cfi_def_cfa_register 6
        .loc 1 11 0
        movl    $0, %eax
        .loc 1 12 0
        leave
.LCFI5:
        .cfi_def_cfa 7, 8
        ret
        .cfi_endproc
.LFE1:
        .size   main, .-main
.Letext0:
        .section        .debug_loc,"",@progbits
.Ldebug_loc0:
.LLST0:
        .quad   .LFB0-.Ltext0
        .quad   .LCFI0-.Ltext0
        .value  0x2
        .byte   0x77
        .sleb128 8
        .quad   .LCFI0-.Ltext0
        .quad   .LCFI1-.Ltext0
        .value  0x2
        .byte   0x77
        .sleb128 16
        .quad   .LCFI1-.Ltext0
        .quad   .LCFI2-.Ltext0
        .value  0x2
        .byte   0x76
        .sleb128 16
        .quad   .LCFI2-.Ltext0
        .quad   .LFE0-.Ltext0
        .value  0x2
        .byte   0x77
        .sleb128 8
        .quad   0x0
        .quad   0x0
.LLST1:
        .quad   .LFB1-.Ltext0
        .quad   .LCFI3-.Ltext0
        .value  0x2
        .byte   0x77
        .sleb128 8
        .quad   .LCFI3-.Ltext0
        .quad   .LCFI4-.Ltext0
        .value  0x2
        .byte   0x77
        .sleb128 16
        .quad   .LCFI4-.Ltext0
        .quad   .LCFI5-.Ltext0
        .value  0x2
        .byte   0x76
        .sleb128 16
        .quad   .LCFI5-.Ltext0
        .quad   .LFE1-.Ltext0
        .value  0x2
        .byte   0x77
        .sleb128 8
        .quad   0x0
        .quad   0x0
        .file 2 "<built-in>"
        .file 3 "/opt/compiler-explorer/gcc-4.5.3/bin/../lib/gcc/x86_64-linux-gnu/4.5.3/include/stdarg.h"
        .section        .debug_info
        .long   0x138
        .value  0x2
        .long   .Ldebug_abbrev0
        .byte   0x8
        .uleb128 0x1
        .long   .LASF9
        .byte   0x4
        .long   .LASF10
        .long   .LASF11
        .quad   .Ltext0
        .quad   .Letext0
        .long   .Ldebug_line0
        .uleb128 0x2
        .long   .LASF6
        .byte   0x3
        .byte   0x28
        .long   0x38
        .uleb128 0x3
        .long   0x4f
        .long   0x48
        .uleb128 0x4
        .long   0x48
        .byte   0x0
        .byte   0x0
        .uleb128 0x5
        .byte   0x8
        .byte   0x7
        .long   .LASF4
        .uleb128 0x6
        .long   .LASF12
        .byte   0x18
        .byte   0x2
        .byte   0x0
        .long   0x94
        .uleb128 0x7
        .long   .LASF0
        .byte   0x2
        .byte   0x0
        .long   0x94
        .byte   0x2
        .byte   0x23
        .uleb128 0x0
        .uleb128 0x7
        .long   .LASF1
        .byte   0x2
        .byte   0x0
        .long   0x94
        .byte   0x2
        .byte   0x23
        .uleb128 0x4
        .uleb128 0x7
        .long   .LASF2
        .byte   0x2
        .byte   0x0
        .long   0x9b
        .byte   0x2
        .byte   0x23
        .uleb128 0x8
        .uleb128 0x7
        .long   .LASF3
        .byte   0x2
        .byte   0x0
        .long   0x9b
        .byte   0x2
        .byte   0x23
        .uleb128 0x10
        .byte   0x0
        .uleb128 0x5
        .byte   0x4
        .byte   0x7
        .long   .LASF5
        .uleb128 0x8
        .byte   0x8
        .uleb128 0x2
        .long   .LASF7
        .byte   0x3
        .byte   0x66
        .long   0x2d
        .uleb128 0x9
        .byte   0x1
        .long   .LASF13
        .byte   0x1
        .byte   0x3
        .long   .LASF14
        .long   0x102
        .quad   .LFB0
        .quad   .LFE0
        .long   .LLST0
        .long   0x102
        .uleb128 0xa
        .long   .LASF15
        .byte   0x1
        .byte   0x3
        .long   0x109
        .byte   0x3
        .byte   0x91
        .sleb128 -232
        .uleb128 0xb
        .uleb128 0xc
        .quad   .LBB2
        .quad   .LBE2
        .uleb128 0xd
        .long   .LASF16
        .byte   0x1
        .byte   0x4
        .long   0x9d
        .byte   0x3
        .byte   0x91
        .sleb128 -224
        .byte   0x0
        .byte   0x0
        .uleb128 0xe
        .byte   0x4
        .byte   0x5
        .string "int"
        .uleb128 0xf
        .byte   0x8
        .long   0x10f
        .uleb128 0x10
        .long   0x114
        .uleb128 0x5
        .byte   0x1
        .byte   0x6
        .long   .LASF8
        .uleb128 0x11
        .byte   0x1
        .long   .LASF17
        .byte   0x1
        .byte   0xa
        .long   0x102
        .quad   .LFB1
        .quad   .LFE1
        .long   .LLST1
        .byte   0x0
        .section        .debug_abbrev
        .uleb128 0x1
        .uleb128 0x11
        .byte   0x1
        .uleb128 0x25
        .uleb128 0xe
        .uleb128 0x13
        .uleb128 0xb
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x1b
        .uleb128 0xe
        .uleb128 0x11
        .uleb128 0x1
        .uleb128 0x12
        .uleb128 0x1
        .uleb128 0x10
        .uleb128 0x6
        .byte   0x0
        .byte   0x0
        .uleb128 0x2
        .uleb128 0x16
        .byte   0x0
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x49
        .uleb128 0x13
        .byte   0x0
        .byte   0x0
        .uleb128 0x3
        .uleb128 0x1
        .byte   0x1
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x1
        .uleb128 0x13
        .byte   0x0
        .byte   0x0
        .uleb128 0x4
        .uleb128 0x21
        .byte   0x0
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x2f
        .uleb128 0xb
        .byte   0x0
        .byte   0x0
        .uleb128 0x5
        .uleb128 0x24
        .byte   0x0
        .uleb128 0xb
        .uleb128 0xb
        .uleb128 0x3e
        .uleb128 0xb
        .uleb128 0x3
        .uleb128 0xe
        .byte   0x0
        .byte   0x0
        .uleb128 0x6
        .uleb128 0x13
        .byte   0x1
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0xb
        .uleb128 0xb
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x1
        .uleb128 0x13
        .byte   0x0
        .byte   0x0
        .uleb128 0x7
        .uleb128 0xd
        .byte   0x0
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x38
        .uleb128 0xa
        .byte   0x0
        .byte   0x0
        .uleb128 0x8
        .uleb128 0xf
        .byte   0x0
        .uleb128 0xb
        .uleb128 0xb
        .byte   0x0
        .byte   0x0
        .uleb128 0x9
        .uleb128 0x2e
        .byte   0x1
        .uleb128 0x3f
        .uleb128 0xc
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x2007
        .uleb128 0xe
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x11
        .uleb128 0x1
        .uleb128 0x12
        .uleb128 0x1
        .uleb128 0x40
        .uleb128 0x6
        .uleb128 0x1
        .uleb128 0x13
        .byte   0x0
        .byte   0x0
        .uleb128 0xa
        .uleb128 0x5
        .byte   0x0
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x2
        .uleb128 0xa
        .byte   0x0
        .byte   0x0
        .uleb128 0xb
        .uleb128 0x18
        .byte   0x0
        .byte   0x0
        .byte   0x0
        .uleb128 0xc
        .uleb128 0xb
        .byte   0x1
        .uleb128 0x11
        .uleb128 0x1
        .uleb128 0x12
        .uleb128 0x1
        .byte   0x0
        .byte   0x0
        .uleb128 0xd
        .uleb128 0x34
        .byte   0x0
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x2
        .uleb128 0xa
        .byte   0x0
        .byte   0x0
        .uleb128 0xe
        .uleb128 0x24
        .byte   0x0
        .uleb128 0xb
        .uleb128 0xb
        .uleb128 0x3e
        .uleb128 0xb
        .uleb128 0x3
        .uleb128 0x8
        .byte   0x0
        .byte   0x0
        .uleb128 0xf
        .uleb128 0xf
        .byte   0x0
        .uleb128 0xb
        .uleb128 0xb
        .uleb128 0x49
        .uleb128 0x13
        .byte   0x0
        .byte   0x0
        .uleb128 0x10
        .uleb128 0x26
        .byte   0x0
        .uleb128 0x49
        .uleb128 0x13
        .byte   0x0
        .byte   0x0
        .uleb128 0x11
        .uleb128 0x2e
        .byte   0x0
        .uleb128 0x3f
        .uleb128 0xc
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x11
        .uleb128 0x1
        .uleb128 0x12
        .uleb128 0x1
        .uleb128 0x40
        .uleb128 0x6
        .byte   0x0
        .byte   0x0
        .byte   0x0
        .section        .debug_pubnames,"",@progbits
        .long   0x24
        .value  0x2
        .long   .Ldebug_info0
        .long   0x13c
        .long   0xa8
        .string "myprintf"
        .long   0x11b
        .string "main"
        .long   0x0
        .section        .debug_pubtypes,"",@progbits
        .long   0x3f
        .value  0x2
        .long   .Ldebug_info0
        .long   0x13c
        .long   0x4f
        .string "__va_list_tag"
        .long   0x2d
        .string "__gnuc_va_list"
        .long   0x9d
        .string "va_list"
        .long   0x0
        .section        .debug_aranges,"",@progbits
        .long   0x2c
        .value  0x2
        .long   .Ldebug_info0
        .byte   0x8
        .byte   0x0
        .value  0x0
        .value  0x0
        .quad   .Ltext0
        .quad   .Letext0-.Ltext0
        .quad   0x0
        .quad   0x0
        .section        .debug_str,"MS",@progbits,1
.LASF5:
        .string "unsigned int"
.LASF12:
        .string "typedef __va_list_tag __va_list_tag"
.LASF6:
        .string "__gnuc_va_list"
.LASF13:
        .string "myprintf"
.LASF0:
        .string "gp_offset"
.LASF1:
        .string "fp_offset"
.LASF7:
        .string "va_list"
.LASF11:
        .string "/home/ubuntu"
.LASF3:
        .string "reg_save_area"
.LASF15:
        .string "format"
.LASF17:
        .string "main"
.LASF4:
        .string "long unsigned int"
.LASF14:
        .string "_Z8myprintfPKcz"
.LASF9:
        .string "GNU C++ 4.5.3"
.LASF16:
        .string "argp"
.LASF10:
        .string "./example.cpp"
.LASF2:
        .string "overflow_arg_area"
.LASF8:
        .string "char"
        .ident  "GCC: (GCC-Explorer-Build) 4.5.3"
        .section        .note.GNU-stack,"",@progbits
