; RUN: llvm-as < %s | opt -reassociate -disable-output

implementation   ; Functions:

int %test(int %A.1, int %B.1, int %C.1, int %D.1) {
	%tmp.16 = and int %A.1, %B.1		; <int> [#uses=1]
	%tmp.18 = and int %tmp.16, %C.1		; <int> [#uses=1]
	%tmp.20 = and int %tmp.18, %D.1		; <int> [#uses=1]
	ret int %tmp.20
}
