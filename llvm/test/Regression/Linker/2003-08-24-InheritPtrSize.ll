; Linking these a module with a specified pointer size to one without a 
; specified pointer size should not cause a warning!

; RUN: llvm-as < %s > Output/%s.out1.bc
; RUN: echo "" | llvm-as > Output/%s.out2.bc
; RUN: llvm-link Output/%s.out[12].bc 2>&1 | not grep WARNING

target pointersize = 64

