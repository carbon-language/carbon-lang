; Test that: extern int X[]  and int X[] = { 1, 2, 3, 4 } are resolved 
; correctly.
;
; RUN: if as < %s | opt -funcresolve > /dev/null
; RUN: then echo "opt ok"
; RUN: else exit 1   # Make sure opt doesn't abort!
; RUN: fi
;
; RUN: if as < %s | opt -funcresolve | dis | grep external
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi
;

%X = external global int
%X = global [4 x int] [ int 1, int 2, int 3, int 4 ]

implementation   ; Functions:

int %foo(int %x) {
bb1:                                    ;[#uses=0]
	store int 5, int* getelementptr (int* %X, long 2)
	%F = getelementptr int* %X, long 2
	%val = load int* %F
        ret int %val
}

