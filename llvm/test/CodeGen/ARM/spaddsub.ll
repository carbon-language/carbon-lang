; RUN: llvm-upgrade < %s | llvm-as | llc -f -march=arm -o %t.s &&
; RUN: not grep "add r13, r13, #0" < %t.s &&
; RUN: not grep "sub r13, r13, #0" < %t.s

int %f() {
entry:
	ret int 1
}


