; This is a basic sanity check for constant propogation.  The add instruction 
; should be eliminated.

; RUN: llvm-as < %s | opt -constprop -die | llvm-dis | not grep phi

int %test(bool %B) {
BB0:
	br bool %B, label %BB1, label %BB3
BB1:
	br label %BB3
BB3:
	%Ret = phi int [1, %BB0], [1, %BB1]
	ret int %Ret
}
