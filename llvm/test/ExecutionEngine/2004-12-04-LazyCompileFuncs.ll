; RUN: llvm-as -f %s -o %t.bc
; RUN: lli -debug-only=jit %t.bc 2>&1 | not grep 'Finished CodeGen of .*Function: F'

%.str_1 = internal constant [7 x sbyte] c"IN F!\0A\00"		; <[7 x sbyte]*> [#uses=1]
%.str_2 = internal constant [7 x sbyte] c"IN G!\0A\00"		; <[7 x sbyte]*> [#uses=1]
%Ptrs = internal constant [2 x void (...)*] [ void (...)* cast (void ()* %F to void (...)*), void (...)* cast (void ()* %G to void (...)*) ]            ; <[2 x void (...)*]*> [#uses=1]

implementation   ; Functions:

declare int %printf(sbyte*, ...)

internal void %F() {
entry:
	%tmp.0 = call int (sbyte*, ...)* %printf( sbyte* getelementptr ([7 x sbyte]* %.str_1, int 0, int 0) )		; <int> [#uses=0]
	ret void
}

internal void %G() {
entry:
	%tmp.0 = call int (sbyte*, ...)* %printf( sbyte* getelementptr ([7 x sbyte]* %.str_2, int 0, int 0) )		; <int> [#uses=0]
	ret void
}

int %main(int %argc, sbyte** %argv) {
entry:
	%tmp.3 = and int %argc, 1		; <int> [#uses=1]
	%tmp.4 = getelementptr [2 x void (...)*]* %Ptrs, int 0, int %tmp.3		; <void (...)**> [#uses=1]
	%tmp.5 = load void (...)** %tmp.4		; <void (...)*> [#uses=1]
	%tmp.5_c = cast void (...)* %tmp.5 to void ()*		; <void ()*> [#uses=1]
	call void %tmp.5_c( )
	ret int undef
}

