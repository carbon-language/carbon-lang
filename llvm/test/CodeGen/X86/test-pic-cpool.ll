; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -relocation-model=pic &&
; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -relocation-model=pic | grep _GLOBAL_OFFSET_TABLE_ &&
; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -relocation-model=pic | grep piclabel | wc -l | grep 3 &&
; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -relocation-model=pic | grep GOTOFF | wc -l | grep 2 &&
; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -relocation-model=pic | grep CPI | wc -l | grep 4

define double %foo(i32 %a.u) {
entry:
    %tmp = icmp eq i32 %a.u,0
    %retval = select i1 %tmp, double 4.561230e+02, double 1.234560e+02
    ret double %retval
}

