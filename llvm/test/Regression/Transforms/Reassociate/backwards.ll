; RUN:  llvm-as < %s | opt -reassociate -dce -gcse | llvm-dis | grep add | wc -l | grep 8
; This test corresponds to GCC PR16157.  Reassociate should arrange for 4 additions to be
; left in each function.

; We currently don't implement this.  This would require looking globally to 
; find out which expressions are available, which we currently do not do.

; XFAIL: *

%a4 = external global int
%a3 = external global int
%a2 = external global int
%a1 = external global int
%a0 = external global int
%b4 = external global int
%b3 = external global int
%b2 = external global int
%b1 = external global int

implementation   ; Functions:

void %test1() {
	%tmp.0 = load int* %a4		; <int> [#uses=1]
	%tmp.1 = load int* %a3		; <int> [#uses=2]
	%tmp.2 = add int %tmp.0, %tmp.1		; <int> [#uses=1]
	%tmp.3 = load int* %a2		; <int> [#uses=3]
	%tmp.4 = add int %tmp.2, %tmp.3		; <int> [#uses=1]
	%tmp.5 = load int* %a1		; <int> [#uses=4]
	%tmp.6 = add int %tmp.4, %tmp.5		; <int> [#uses=1]
	%tmp.7 = load int* %a0		; <int> [#uses=4]
	%tmp.8 = add int %tmp.6, %tmp.7		; <int> [#uses=1]
	store int %tmp.8, int* %a4
	%tmp.11 = add int %tmp.1, %tmp.3		; <int> [#uses=1]
	%tmp.13 = add int %tmp.11, %tmp.5		; <int> [#uses=1]
	%tmp.15 = add int %tmp.13, %tmp.7		; <int> [#uses=1]
	store int %tmp.15, int* %a3
	%tmp.18 = add int %tmp.3, %tmp.5		; <int> [#uses=1]
	%tmp.20 = add int %tmp.18, %tmp.7		; <int> [#uses=1]
	store int %tmp.20, int* %a2
	%tmp.23 = add int %tmp.5, %tmp.7		; <int> [#uses=1]
	store int %tmp.23, int* %a1
	ret void
}

void %test2() {
	%tmp.0 = load int* %a4		; <int> [#uses=1]
	%tmp.1 = load int* %a3		; <int> [#uses=2]
	%tmp.2 = add int %tmp.0, %tmp.1		; <int> [#uses=1]
	%tmp.3 = load int* %a2		; <int> [#uses=3]
	%tmp.4 = add int %tmp.2, %tmp.3		; <int> [#uses=1]
	%tmp.5 = load int* %a1		; <int> [#uses=4]
	%tmp.6 = add int %tmp.4, %tmp.5		; <int> [#uses=1]
	%tmp.7 = load int* %a0		; <int> [#uses=4]
	%tmp.8 = add int %tmp.6, %tmp.7		; <int> [#uses=1]
	store int %tmp.8, int* %b4
	%tmp.11 = add int %tmp.1, %tmp.3		; <int> [#uses=1]
	%tmp.13 = add int %tmp.11, %tmp.5		; <int> [#uses=1]
	%tmp.15 = add int %tmp.13, %tmp.7		; <int> [#uses=1]
	store int %tmp.15, int* %b3
	%tmp.18 = add int %tmp.3, %tmp.5		; <int> [#uses=1]
	%tmp.20 = add int %tmp.18, %tmp.7		; <int> [#uses=1]
	store int %tmp.20, int* %b2
	%tmp.23 = add int %tmp.5, %tmp.7		; <int> [#uses=1]
	store int %tmp.23, int* %b1
	ret void
}
