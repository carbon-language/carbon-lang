; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm &&
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=arm-linux | grep mov | wc -l | grep 1 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=arm-apple-darwin | grep mov | wc -l | grep 2

%str = internal constant [12 x sbyte] c"Hello World\00"

int %main() {
	%tmp = call int %puts( sbyte* getelementptr ([12 x sbyte]* %str, int 0, uint 0) )
	ret int 0
}

declare int %puts(sbyte*)
