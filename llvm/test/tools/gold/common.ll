; RUN: llvm-as %s -o %t1.o
; RUN: llvm-as %p/Inputs/common.ll -o %t2.o

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t1.o %t2.o -o %t3.o
; RUN: llvm-dis %t3.o -o - | FileCheck %s

@a = common global i8 0, align 8

; Shared library case, we merge @a as common and keep it for the symbol table.
; CHECK: @a = common global i16 0, align 8

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    %t1.o %t2.o -o %t3.o
; RUN: llvm-dis %t3.o -o - | FileCheck --check-prefix=EXEC %s

; All IR case, we internalize a after merging.
; EXEC: @a = internal global i16 0, align 8

; RUN: llc %p/Inputs/common.ll -o %t2.o -filetype=obj
; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    %t1.o %t2.o -o %t3.o
; RUN: llvm-dis %t3.o -o - | FileCheck --check-prefix=MIXED %s

; Mixed ELF and IR. We keep ours as common so the linker will finish the merge.
; MIXED: @a = common global i8 0, align 8
