; This test checks to make sure phi nodes are updated properly
;
; RUN: as < %s | opt -tailduplicate -disable-output



int %test(bool %c, int %X, int %Y) {
	br label %L

L:
	%A = add int %X, %Y
	br bool %c, label %T, label %F

F:
	br bool %c, label %L, label %T

T:
	%V = phi int [%A, %L], [0, %F]
	ret int %V
}
