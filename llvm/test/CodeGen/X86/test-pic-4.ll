; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -relocation-model=pic &&
; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -relocation-model=pic | grep _GLOBAL_OFFSET_TABLE_ &&
; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -relocation-model=pic | grep piclabel | wc -l | grep 3 &&
; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -relocation-model=pic | grep PLT | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -relocation-model=pic | grep "GOT" | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -relocation-model=pic | grep "GOTOFF" | wc -l | grep 0

@pfoo = external global void(...)* 

define void @bar() {
entry:
    %tmp = call void(...)*(...)* @afoo()
    store void(...)* %tmp, void(...)** @pfoo
    %tmp1 = load void(...)** @pfoo
    call void(...)* %tmp1()
    br label %return
return:
    ret void
}

declare void(...)* @afoo(...)
