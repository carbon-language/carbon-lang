;;===-- X86CompilationCallback_Win64.asm - Implement Win64 JIT callback ---===
;;
;;                     The LLVM Compiler Infrastructure
;;
;; This file is distributed under the University of Illinois Open Source
;; License. See LICENSE.TXT for details.
;;
;;===----------------------------------------------------------------------===
;;
;; This file implements the JIT interfaces for the X86 target.
;;
;;===----------------------------------------------------------------------===

extrn X86CompilationCallback2: PROC

.code
X86CompilationCallback proc
    ; Save all int arg registers into register spill area.
    mov     [rsp+ 8], rcx
    mov     [rsp+16], rdx
    mov     [rsp+24], r8
    mov     [rsp+32], r9

    push    rbp

    ; Save RSP.
    mov     rbp, rsp

    ; Align stack on 16-byte boundary.
    and     rsp, -16

    ; Save all XMM arg registers. Also allocate reg spill area.
    sub     rsp, 96
    movaps  [rsp   +32],  xmm0
    movaps  [rsp+16+32],  xmm1
    movaps  [rsp+32+32],  xmm2
    movaps  [rsp+48+32],  xmm3

    ; JIT callee

    ; Pass prev frame and return address.
    mov     rcx, rbp
    mov     rdx, qword ptr [rbp+8]
    call    X86CompilationCallback2

    ; Restore all XMM arg registers.
    movaps  xmm3, [rsp+48+32]
    movaps  xmm2, [rsp+32+32]
    movaps  xmm1, [rsp+16+32]
    movaps  xmm0, [rsp   +32]

    ; Restore RSP.
    mov     rsp, rbp

    ; Restore RBP.
    pop     rbp

    ; Restore all int arg registers.
    mov     r9,  [rsp+32]
    mov     r8,  [rsp+24]
    mov     rdx, [rsp+16]
    mov     rcx, [rsp+ 8]

    ret
X86CompilationCallback endp

End
