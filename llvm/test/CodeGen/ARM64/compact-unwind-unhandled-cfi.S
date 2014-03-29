; RUN: llvm-mc -triple arm64-apple-darwin -filetype=obj -o /dev/null %s

        .text
        .globl _foo
        .cfi_startproc
_foo:
        stp x29, x30, [sp, #-16]!
 .cfi_adjust_cfa_offset 16

        ldp x29, x30, [sp], #16
 .cfi_adjust_cfa_offset -16
        .cfi_restore x29
        .cfi_restore x30

        ret

        .cfi_endproc
