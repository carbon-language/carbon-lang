; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -relocation-model=pic &&
; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -relocation-model=pic | grep _GLOBAL_OFFSET_TABLE_ &&
; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -relocation-model=pic | grep piclabel | wc -l | grep 3 &&
; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -relocation-model=pic | grep PLT | wc -l | grep 1

define void %bar() {
entry:
    call void(...)* %foo()
    br label %return
return:
    ret void
}

declare void %foo(...)
