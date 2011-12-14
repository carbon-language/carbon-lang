# RUN: llvm-mc -triple i386-unknown-unknown %s -I %p | FileCheck %s

.data
.incbin "incbin_abcd"

# CHECK: .byte	97
# CHECK: .byte	98
# CHECK: .byte	99
# CHECK: .byte	100
