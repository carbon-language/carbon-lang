; This testcase ensures that redundant loads are preserved when they are not 
; allowed to be eliminated.
; RUN: as < %s | dis > Output/%s.before
; RUN: as < %s | opt -load-vn -gcse | dis > Output/%s.after
; RUN: echo some output
; RUN: diff Output/%s.before Output/%s.after
;
int "test1"(int* %P) {
	%A = load int* %P
	store int 1, int * %P
	%B = load int* %P
	%C = add int %A, %B
	ret int %C
}

int "test2"(int* %P) {
	%A = load int* %P
	br label %BB2
BB2:
	store int 5, int * %P
	br label %BB3
BB3:
	%B = load int* %P
	%C = add int %A, %B
	ret int %C
}


