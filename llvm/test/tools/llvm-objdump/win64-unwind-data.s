// This test checks that the unwind data is dumped by llvm-objdump.
// RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj %s | llvm-objdump -u - | FileCheck %s

// CHECK:      Unwind info:
// CHECK:      Function Table:
// CHECK-NEXT: Start Address: .text
// CHECK-NEXT: End Address: .text + 0x001b
// CHECK-NEXT: Unwind Info Address: .xdata
// CHECK-NEXT: Version: 1
// CHECK-NEXT: Flags: 1 UNW_ExceptionHandler
// CHECK-NEXT: Size of prolog: 18
// CHECK-NEXT: Number of Codes: 8
// CHECK-NEXT: Frame register: RBX
// CHECK-NEXT: Frame offset: 0
// CHECK-NEXT: Unwind Codes:
// CHECK-NEXT: 0x00: UOP_SetFPReg
// CHECK-NEXT: 0x0f: UOP_PushNonVol RBX
// CHECK-NEXT: 0x0e: UOP_SaveXMM128 XMM8 [0x0000]
// CHECK-NEXT: 0x09: UOP_SaveNonVol RSI [0x0010]
// CHECK-NEXT: 0x04: UOP_AllocSmall 24
// CHECK-NEXT: 0x00: UOP_PushMachFrame w/o error code
// CHECK:      Function Table:
// CHECK-NEXT: Start Address: .text + 0x0012
// CHECK-NEXT: End Address: .text + 0x0012
// CHECK-NEXT: Unwind Info Address: .xdata + 0x001c
// CHECK-NEXT: Version: 1
// CHECK-NEXT: Flags: 4 UNW_ChainInfo
// CHECK-NEXT: Size of prolog: 0
// CHECK-NEXT: Number of Codes: 0
// CHECK-NEXT: No frame pointer used
// CHECK:      Function Table:
// CHECK-NEXT: Start Address: .text + 0x001b
// CHECK-NEXT: End Address: .text + 0x001c
// CHECK-NEXT: Unwind Info Address: .xdata + 0x002c
// CHECK-NEXT: Version: 1
// CHECK-NEXT: Flags: 0
// CHECK-NEXT: Size of prolog: 0
// CHECK-NEXT: Number of Codes: 0
// CHECK-NEXT: No frame pointer used
// CHECK:      Function Table:
// CHECK-NEXT: Start Address: .text + 0x001c
// CHECK-NEXT: End Address: .text + 0x0039
// CHECK-NEXT: Unwind Info Address: .xdata + 0x0034
// CHECK-NEXT: Version: 1
// CHECK-NEXT: Flags: 0
// CHECK-NEXT: Size of prolog: 14
// CHECK-NEXT: Number of Codes: 6
// CHECK-NEXT: No frame pointer used
// CHECK-NEXT: Unwind Codes:
// CHECK-NEXT: 0x0e: UOP_AllocLarge 8454128
// CHECK-NEXT: 0x07: UOP_AllocLarge 8190
// CHECK-NEXT: 0x00: UOP_PushMachFrame w/o error code

    .text
    .globl func
    .def func; .scl 2; .type 32; .endef
    .seh_proc func
func:
    .seh_pushframe @code
    subq $24, %rsp
    .seh_stackalloc 24
    movq %rsi, 16(%rsp)
    .seh_savereg %rsi, 16
    movups %xmm8, (%rsp)
    .seh_savexmm %xmm8, 0
    pushq %rbx
    .seh_pushreg 3
    mov %rsp, %rbx
    .seh_setframe 3, 0
    .seh_endprologue
    .seh_handler __C_specific_handler, @except
    .seh_handlerdata
    .long 0
    .text
    .seh_startchained
    .seh_endprologue
    .seh_endchained
    lea (%rbx), %rsp
    pop %rbx
    addq $24, %rsp
    ret
    .seh_endproc

// Test emission of small functions.
    .globl smallFunc
    .def smallFunc; .scl 2; .type 32; .endef
    .seh_proc smallFunc
smallFunc:
    ret
    .seh_endproc

// Function with big stack allocation.
    .globl smallFunc
    .def allocFunc; .scl 2; .type 32; .endef
    .seh_proc smallFunc
allocFunc:
    .seh_pushframe @code
    subq $65520, %rsp
    .seh_stackalloc 65520
    sub $8454128, %rsp
    .seh_stackalloc 8454128
    .seh_endprologue
    add $8454128, %rsp
    addq $65520, %rsp
    ret
    .seh_endproc
