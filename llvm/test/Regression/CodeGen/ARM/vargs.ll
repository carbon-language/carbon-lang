; RUN: llvm-as < %s | llc -march=arm
%str = internal constant [43 x sbyte] c"Hello World %d %d %d %d %d %d %d %d %d %d\0A\00"		; <[43 x sbyte]*> [#uses=1]

implementation   ; Functions:

int %main() {
entry:
	%tmp = call int (sbyte*, ...)* %printf( sbyte* getelementptr ([43 x sbyte]* %str, int 0, uint 0), int 1, int 2, int 3, int 4, int 5, int 6, int 7, int 8, int 9, int 10 )		; <int> [#uses=0]
	%tmp2 = call int (sbyte*, ...)* %printf( sbyte* getelementptr ([43 x sbyte]* %str, int 0, uint 0), int 10, int 9, int 8, int 7, int 6, int 5, int 4, int 3, int 2, int 1 )		; <int> [#uses=0]
	ret int 11
}

declare int %printf(sbyte*, ...)
