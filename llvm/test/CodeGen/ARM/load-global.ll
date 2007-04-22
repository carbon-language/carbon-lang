; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -mtriple=arm-apple-darwin -relocation-model=dynamic-no-pic | \
; RUN:   grep {L_G\$non_lazy_ptr} | wc -l | grep 2
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -mtriple=arm-apple-darwin -relocation-model=pic | \
; RUN:   grep {ldr.*pc} | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -mtriple=arm-linux-gnueabi -relocation-model=pic | \
; RUN:   grep {GOT} | wc -l | grep 1

%G = external global int

int %test1() {
	%tmp = load int* %G
	ret int %tmp
}
