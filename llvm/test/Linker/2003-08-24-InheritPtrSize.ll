; Linking these a module with a specified pointer size to one without a 
; specified pointer size should not cause a warning!

; RUN: llvm-as < %s > %t.out1.bc
; RUN: echo "" | llvm-as > %t.out2.bc
; RUN: llvm-link %t.out[12].bc 2>&1 | not grep WARNING

target pointersize = 64

