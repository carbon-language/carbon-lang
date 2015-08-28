; REQUIRES: ld_emu_elf32ppc

; RUN: llvm-as %s -o %t.o
; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so -m elf32ppc \
; RUN:    -plugin-opt=mtriple=powerpc-linux-gnu \
; RUN:    -plugin-opt=obj-path=%t3.o \
; RUN:    -shared %t.o -o %t2
; RUN: llvm-readobj --file-headers %t2 | FileCheck  --check-prefix=DSO %s
; RUN: llvm-readobj --file-headers %t3.o | FileCheck --check-prefix=REL %s

; REL:       Type: Relocatable
; REL-NEXT:  Machine: EM_PPC

; DSO:       Type: SharedObject
; DSO-NEXT:  Machine: EM_PPC
