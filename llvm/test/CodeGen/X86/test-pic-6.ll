; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -relocation-model=pic &&
; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -relocation-model=pic | grep _GLOBAL_OFFSET_TABLE_ &&
; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -relocation-model=pic | grep piclabel | wc -l | grep 3 &&
; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -relocation-model=pic | grep GOT | wc -l | grep 3

%ptr = global i32* null
%dst = global i32 0
%src = global i32 0

define void %foo() {
entry:
    store i32* %dst, i32** %ptr
    %tmp.s = load i32* %src
    store i32 %tmp.s, i32* %dst
    ret void
}

