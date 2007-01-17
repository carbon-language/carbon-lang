; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep moveq | wc -l | grep 1 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep movgt | wc -l | grep 1 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep movlt | wc -l | grep 1 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep movle | wc -l | grep 1 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep movls | wc -l | grep 1 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep movhi | wc -l | grep 1

int %f1(int %a) {
entry:
	%tmp = seteq int %a, 4		; <bool> [#uses=1]
	%tmp1 = select bool %tmp, int 2, int 3
	ret int %tmp1
}

int %f2(int %a) {
entry:
	%tmp = setgt int %a, 4		; <bool> [#uses=1]
	%tmp1 = select bool %tmp, int 2, int 3
	ret int %tmp1
}

int %f3(int %a, int %b) {
entry:
	%tmp = setlt int %a, %b		; <bool> [#uses=1]
	%tmp1 = select bool %tmp, int 2, int 3
	ret int %tmp1
}

int %f4(int %a, int %b) {
entry:
	%tmp = setle int %a, %b		; <bool> [#uses=1]
	%tmp1 = select bool %tmp, int 2, int 3
	ret int %tmp1
}

int %f5(uint %a, uint %b) {
entry:
	%tmp = setle uint %a, %b		; <bool> [#uses=1]
	%tmp1 = select bool %tmp, int 2, int 3
	ret int %tmp1
}

int %f6(uint %a, uint %b) {
entry:
	%tmp = setgt uint %a, %b		; <bool> [#uses=1]
	%tmp1 = select bool %tmp, int 2, int 3
	ret int %tmp1
}
