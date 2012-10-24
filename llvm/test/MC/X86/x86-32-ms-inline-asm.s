// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

.intel_syntax
mov eax, [ebx].0
mov [ebx].4, ecx
.att_syntax
        
; CHECK: movl (%ebx), %eax
; CHECK: encoding: [0x8b,0x03]
; CHECK: movl %ecx, 4(%ebx)
; CHECK: encoding: [0x89,0x4b,0x04]
        
