; The %A getelementptr instruction should be eliminated here

; RUN: if as < %s | opt -instcombine -die | dis | grep getelementptr | grep '%A '
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

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

sbyte * %foo5() {
	; This should be turned into a constexpr instead of being an instruction
	%A = getelementptr [10 x sbyte]* %Global, long 0, long 4
	ret sbyte* %A
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
