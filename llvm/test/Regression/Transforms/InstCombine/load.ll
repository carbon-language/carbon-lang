; This test makes sure that these instructions are properly eliminated.
;

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep load

%X = constant int 42
%Y = constant [2 x { int, float }] [ { int, float } { int 12, float 1.0 }, 
                                     { int, float } { int 37, float 1.2312 } ]
%Z = constant [2 x { int, float }] zeroinitializer

int %test1() {
	%B = load int* %X
	ret int %B
}

float %test2() {
	%A = getelementptr [2 x { int, float}]* %Y, long 0, long 1, ubyte 1
	%B = load float* %A
	ret float %B
}


int %test3() {
	%A = getelementptr [2 x { int, float}]* %Y, long 0, long 0, ubyte 0
	%B = load int* %A
	ret int %B
}

int %test4() {
	%A = getelementptr [2 x { int, float}]* %Z, long 0, long 1, ubyte 0
	%B = load int* %A
	ret int %B
}
