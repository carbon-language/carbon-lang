; It is illegal to remove BB1 because it will mess up the PHI node!
;
; RUN: llvm-as < %s | opt -adce | llvm-dis | grep BB1


int "test"(bool %C, int %A, int %B) {
	br bool %C, label %BB1, label %BB2
BB1:
	br label %BB2
BB2:
	%R = phi int [%A, %0], [%B, %BB1]
	ret int %R
}
