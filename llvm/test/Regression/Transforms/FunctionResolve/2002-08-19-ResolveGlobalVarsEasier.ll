; Test that: extern int X[]  and int X[] = { 1, 2, 3, 4 } are resolved 
; correctly.  This doesn't have constantexprs
;
; RUN: llvm-as < %s | opt -funcresolve | llvm-dis | not grep external
;

%X = external global [0 x int]
%X = global [4 x int] [ int 1, int 2, int 3, int 4 ]

implementation   ; Functions:

int %foo(int %x) {
bb1:                                    ;[#uses=0]
	%G = getelementptr [0 x int]* %X, long 0, long 1
	store int 5, int* %G
	%F = getelementptr [0 x int]* %X, long 0, long 2
	%val = load int* %F
        ret int %val
}

