; This testcase ensures that redundant loads are eliminated when they should 
; be.  All RL variables (redundant loads) should be eliminated.
;
; RUN: if as < %s | opt -load-vn -gcse | dis | grep %RL
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi
;
int "test1"(int* %P) {
	%A = load int* %P
	%RL = load int* %P
	%C = add int %A, %RL
	ret int %C
}

int "test2"(int* %P) {
	%A = load int* %P
	br label %BB2
BB2:
	br label %BB3
BB3:
	%RL = load int* %P
	%B = add int %A, %RL
	ret int %B
}

