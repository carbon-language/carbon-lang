; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | \
; RUN:   grep ldmia | wc -l | grep 2
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | \
; RUN:   grep ldmib | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=arm-apple-darwin | \
; RUN:   grep {ldmfd sp\!} | wc -l | grep 3

%X = external global [0 x int]

int %t1() {
	%tmp = load int* getelementptr ([0 x int]* %X, int 0, int 0)
	%tmp3 = load int* getelementptr ([0 x int]* %X, int 0, int 1)
	%tmp4 = tail call int %f1( int %tmp, int %tmp3 )
	ret int %tmp4
}

int %t2() {
	%tmp = load int* getelementptr ([0 x int]* %X, int 0, int 2)
	%tmp3 = load int* getelementptr ([0 x int]* %X, int 0, int 3)
	%tmp5 = load int* getelementptr ([0 x int]* %X, int 0, int 4)
	%tmp6 = tail call int %f2( int %tmp, int %tmp3, int %tmp5 )
	ret int %tmp6
}

int %t3() {
	%tmp = load int* getelementptr ([0 x int]* %X, int 0, int 1)
	%tmp3 = load int* getelementptr ([0 x int]* %X, int 0, int 2)
	%tmp5 = load int* getelementptr ([0 x int]* %X, int 0, int 3)
	%tmp6 = tail call int %f2( int %tmp, int %tmp3, int %tmp5 )
	ret int %tmp6
}

declare int %f1(int, int)
declare int %f2(int, int, int)
