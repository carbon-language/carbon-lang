; The %A getelementptr instruction should be eliminated here

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep -v '%B' | not grep getelementptr

%Global = constant [10 x sbyte] c"helloworld"

implementation

int *%foo1(int* %I) { ; Test noop elimination
	%A = getelementptr int* %I, long 0
	ret int * %A
}

int* %foo2(int* %I) {  ; Test noop elimination
	%A = getelementptr int* %I
	ret int* %A
}
int* %foo3(int * %I) { ; Test that two array indexing geps fold
	%A = getelementptr int* %I, long 17
	%B = getelementptr int* %A, long 4
	ret int* %B
}

int* %foo4({int} *%I) { ; Test that two getelementptr insts fold
	%A = getelementptr {int}* %I, long 1
	%B = getelementptr {int}* %A, long 0, ubyte 0
	ret int* %B
}

void %foo5(sbyte %B) {
	; This should be turned into a constexpr instead of being an instruction
	%A = getelementptr [10 x sbyte]* %Global, long 0, long 4
	store sbyte %B, sbyte* %A
	ret void
}

int* %foo6() {
	%M = malloc [4 x int]
	%A = getelementptr [4 x int]* %M, long 0, long 0
	%B = getelementptr int* %A, long 2
	ret int* %B
}

int* %foo7(int* %I, long %C, long %D) {
	%A = getelementptr int* %I, long %C
	%B = getelementptr int* %A, long %D
	ret int* %B
}

sbyte* %foo8([10 x int]* %X) {
	%A = getelementptr [10 x int]* %X, long 0, long 0   ;; Fold into the cast.
	%B = cast int* %A to sbyte*
	ret sbyte * %B
}

int %test9() {
	%A = getelementptr {int, double}* null, int 0, uint 1
	%B = cast double* %A to int
	ret int %B
}

bool %test10({int, int} * %x, {int, int} * %y) {
        %tmp.1 = getelementptr {int,int}* %x, int 0, uint 1
        %tmp.3 = getelementptr {int,int}* %y, int 0, uint 1
        %tmp.4 = seteq int* %tmp.1, %tmp.3    ;; seteq x, y
        ret bool %tmp.4
}

