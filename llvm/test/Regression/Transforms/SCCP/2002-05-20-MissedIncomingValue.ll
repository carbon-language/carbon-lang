; This test shows a case where SCCP is incorrectly eliminating the PHI node
; because it thinks it has a constant 0 value, when it really doesn't.

; RUN: llvm-as < %s | opt -sccp | llvm-dis | grep phi

int "test"(int %A, bool %c) {
bb1:
	br label %BB2
BB2:
	%V = phi int [0, %bb1], [%A, %BB4]
	br label %BB3

BB3:
	br bool %c, label %BB4, label %BB5
BB4:
	br label %BB2

BB5:
	ret int %V
}
