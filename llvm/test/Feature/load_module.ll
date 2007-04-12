; PR1318
; RUN: llvm-as < %s > %t.bc &&
; RUN: opt -load=%llvmlibsdir/LLVMHello%shlibext -hello \
; RUN:   -disable-output %t.bc 2>&1 | grep Hello

@junk = global i32 0

define i32* @somefunk() {
  ret i32* @junk
}

