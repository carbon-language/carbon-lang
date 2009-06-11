; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep store
; PR4366

define void @a() {
  store i32 0, i32 addrspace(1)* null
  ret void
}
