; This test makes sure that these instructions are properly eliminated.
;

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep load

%X = constant int 42
%X2 = constant int 47
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

; load (select (Cond, &V1, &V2))  --> select(Cond, load &V1, load &V2)
int %test5(bool %C) {
	%Y = select bool %C, int* %X, int* %X2
	%Z = load int* %Y
	ret int %Z
}

; load (phi (&V1, &V2, &V3))  --> phi(load &V1, load &V2, load &V3)
int %test6(bool %C) {
entry:
        br bool %C, label %cond_true.i, label %cond_continue.i

cond_true.i:
        br label %cond_continue.i

cond_continue.i:
        %mem_tmp.i.0 = phi int* [ %X, %cond_true.i ], [ %X2, %entry ]
        %tmp.3 = load int* %mem_tmp.i.0
        ret int %tmp.3
}

int %test7(int %X) {
	%V = getelementptr int* null, int %X
	%R = load int* %V
	ret int %R
}
