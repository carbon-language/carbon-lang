; RUN: llvm-as %s -o %t.o
; RUN: ld -plugin %llvmshlibdir/LLVMgold.so -m elf32ppc \
; RUN:    -plugin-opt=mtriple=powerpc-linux-gnu \
; RUN:    -shared %t.o -o %t2.o
; RUN: llvm-readobj %t2.o | FileCheck %s

; CHECK: Format: ELF32-ppc
