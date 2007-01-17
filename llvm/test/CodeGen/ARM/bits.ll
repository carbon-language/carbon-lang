; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep and      | wc -l | grep 1 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep orr      | wc -l | grep 1 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep eor      | wc -l | grep 1 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep mov.*lsl | wc -l | grep 1 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep mov.*asr | wc -l | grep 1

int %f1(int %a, int %b) {
entry:
	%tmp2 = and int %b, %a		; <int> [#uses=1]
	ret int %tmp2
}

int %f2(int %a, int %b) {
entry:
	%tmp2 = or int %b, %a		; <int> [#uses=1]
	ret int %tmp2
}

int %f3(int %a, int %b) {
entry:
	%tmp2 = xor int %b, %a		; <int> [#uses=1]
	ret int %tmp2
}

int %f4(int %a, ubyte %b) {
entry:
	%tmp3 = shl int %a, ubyte %b		; <int> [#uses=1]
	ret int %tmp3
}

int %f5(int %a, ubyte %b) {
entry:
	%tmp3 = shr int %a, ubyte %b		; <int> [#uses=1]
	ret int %tmp3
}
