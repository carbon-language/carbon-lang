; RUN: llvm-as < %s | opt -tailduplicate  | llvm-dis | grep add | not grep uses=1

int %test1(bool %C, int %A, int* %P) {
entry:
        br bool %C, label %L1, label %L2

L1:
	store int 1, int* %P
        br label %L2

L2:
	%X = add int %A, 17
	ret int %X
}

int %test2(bool %C, int %A, int* %P) {
entry:
        br bool %C, label %L1, label %L2

L1:
	store int 1, int* %P
        br label %L3

L2:
	store int 7, int* %P
	br label %L3
L3:
	%X = add int %A, 17
	ret int %X
}

