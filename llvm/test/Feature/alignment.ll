; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

%X = global int 4, align 16

int *%test() align 32 {
	%X = alloca int, align 4
	%Y = alloca int, uint 42, align 16
	%Z = alloca int, align 0
	ret int *%X
}

int *%test2() {
	%X = malloc int, align 4
	%Y = malloc int, uint 42, align 16
	%Z = malloc int, align 0
	ret int *%X
}
