; The %A getelementptr instruction should be eliminated here

; RUN: if as < %s | opt -instcombine -die | dis | grep getelementptr | grep '%A'
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

%Global = constant [10 x sbyte] c"helloworld"

implementation

int *"foo1"(int * %I) { ; Test noop elimination
	%A = getelementptr int* %I, uint 0
	ret int * %A
}

int* %foo2(int* %I) {  ; Test noop elimination
	%A = getelementptr int* %I
	ret int* %A
}
int* %foo3(int * %I) { ; Test that two array indexing geps fold
	%A = getelementptr int* %I, uint 17
	%B = getelementptr int* %A, uint 4
	ret int* %B
}

int* %foo4({int} *%I) { ; Test that two getelementptr insts fold
	%A = getelementptr {int}* %I, uint 1
	%B = getelementptr {int}* %A, uint 0, ubyte 0
	ret int* %B
}

sbyte * %foo5() {
	; This should be turned into a constexpr instead of being an instruction
	%A = getelementptr [10 x sbyte]* %Global, uint 0, uint 4
	ret sbyte* %A
}
