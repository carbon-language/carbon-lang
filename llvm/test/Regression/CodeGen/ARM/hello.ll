; RUN: llvm-as < %s | llc -march=arm &&
; RUN: llvm-as < %s | llc -march=arm | grep mov | wc -l | grep 1
%str = internal constant [12 x sbyte] c"Hello World\00"		; <[12 x sbyte]*> [#uses=1]

implementation   ; Functions:

int %main() {
entry:
	%tmp = call int %puts( sbyte* getelementptr ([12 x sbyte]* %str, int 0, uint 0) )		; <int> [#uses=0]
	ret int 0
}

declare int %puts(sbyte*)
