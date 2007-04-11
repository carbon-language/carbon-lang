; PR1318
; RUN: llvm-as < %s > %t.bc &&
; RUN: opt -load=%llvmlibsdir/LLVMHello%shlibext -hello \
; RUN:   -disable-output %t.bc | grep Hello

@junk = global i32 0
