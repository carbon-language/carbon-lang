extrn X86CompilationCallback2: PROC

.code
X86CompilationCallback proc
    push    rbp

    ; Save RSP
    mov     rbp, rsp

    ; Save all int arg registers
    push    rcx
    push    rdx
    push    r8
    push    r9

    ; Align stack on 16-byte boundary.
    and     rsp, -16

    ; Save all XMM arg registers
    sub     rsp, 64
    movaps  [rsp],     xmm0
    movaps  [rsp+16],  xmm1
    movaps  [rsp+32],  xmm2
    movaps  [rsp+48],  xmm3

    ; JIT callee

    ; Pass prev frame and return address
    mov     rcx, rbp
    mov     rdx, qword ptr [rbp+8]
    call    X86CompilationCallback2

    ; Restore all XMM arg registers
    movaps  xmm3, [rsp+48]
    movaps  xmm2, [rsp+32]
    movaps  xmm1, [rsp+16]
    movaps  xmm0, [rsp]

    ; Restore RSP
    mov     rsp, rbp

    ; Restore all int arg registers
    sub     rsp, 32
    pop     r9
    pop     r8
    pop     rdx
    pop     rcx

    ; Restore RBP
    pop     rbp
    ret
X86CompilationCallback endp

End
