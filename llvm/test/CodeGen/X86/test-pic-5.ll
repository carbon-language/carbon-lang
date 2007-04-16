; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -relocation-model=pic \
; RUN:   -o %t -f
; RUN: grep _GLOBAL_OFFSET_TABLE_ %t
; RUN: grep piclabel %t | wc -l | grep 3
; RUN: grep PLT %t | wc -l | grep 1

@ptr = external global i32* 

define void @foo() {
entry:
    %ptr = malloc i32, i32 10
    ret void
}

