; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm &&
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=arm-apple-darwin -relocation-model=dynamic-no-pic | grep "L_G$non_lazy_ptr" | wc -l | grep 2 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=arm-apple-darwin -relocation-model=pic | grep "ldr.*pc" | wc -l | grep 1

%G = external global int

int %test1() {
	%tmp = load int* %G
	ret int %tmp
}
