; This testcase ensures that redundant loads are preserved when they are not 
; allowed to be eliminated.
; RUN: llvm-as < %s | opt -load-vn -gcse | llvm-dis | grep sub
;
int %test1(int* %P) {
	%A = load int* %P
	store int 1, int* %P
	%B = load int* %P
	%C = sub int %A, %B
	ret int %C
}

int %test2(int* %P) {
	%A = load int* %P
	br label %BB2
BB2:
	store int 5, int* %P
	br label %BB3
BB3:
	%B = load int* %P
	%C = sub int %A, %B
	ret int %C
}


