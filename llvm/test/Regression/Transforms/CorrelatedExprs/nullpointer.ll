; a load or store of a pointer indicates that the pointer is not null.
; Any succeeding uses of the pointer should get this info

; RUN: llvm-as < %s | opt -cee -instcombine -simplifycfg | llvm-dis | not grep br

implementation   ; Functions:

int %nullptr(int* %j) {
bb0:
	store int 7, int* %j               ; j != null
	%cond220 = seteq int* %j, null     ; F
	br bool %cond220, label %bb3, label %bb4  ; direct branch

bb3:
	ret int 4                          ; Dead code
bb4:
	ret int 3                          ; Live code
}
