; This tests to make sure that G ends up in the globals graph of the BU pass.
; If it is not, then %G will get converted to a 'constant' from a 'global'
;
; RUN: as < %s | opt -ds-opt -globaldce | dis | grep %G


%G = internal global int 0		; <int*> [#uses=2]

implementation   ; Functions:

internal void %foo() {
	%tmp.0 = load int* %G		; <int> [#uses=1]
	%tmp.1 = add int %tmp.0, 1		; <int> [#uses=1]
	store int %tmp.1, int* %G
	ret void
}

int %main() {
	call void %foo( )
	ret int 0
}
