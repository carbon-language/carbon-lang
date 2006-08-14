; RUN: llvm-as < %s | llc -march=arm &&
; RUN: llvm-as < %s | llc -march=arm | grep "ldr r0.*#0" | wc -l | grep 2 &&
; RUN: llvm-as < %s | llc -march=arm | grep "ldr r0.*#4092" | wc -l | grep 1

int %f1(int* %v) {
entry:
	%tmp = load int* %v		; <int> [#uses=1]
	ret int %tmp
}

int %f2(int* %v) {
entry:
	%tmp2 = getelementptr int* %v, int 1023		; <int*> [#uses=1]
	%tmp = load int* %tmp2		; <int> [#uses=1]
	ret int %tmp
}

int %f3(int* %v) {
entry:
	%tmp2 = getelementptr int* %v, int 1024		; <int*> [#uses=1]
	%tmp = load int* %tmp2		; <int> [#uses=1]
	ret int %tmp
}
