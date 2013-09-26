# RUN: llvm-mc -filetype=asm -triple x86_64-pc-linux-gnu <%s | FileCheck %s

# Should use SPARC as the target to test this. However, SPARC does not support
# asm parsing yet.

# CHECK: .cfi_window_save


f:
        .cfi_startproc
        nop
        .cfi_window_save
        nop
        .cfi_endproc

