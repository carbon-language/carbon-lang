; This testcase ensures that redundant loads are eliminated when they should 
; be.  All RL variables (redundant loads) should be eliminated.
;
; RUN: as < %s | opt -load-vn -gcse | dis | not grep %RL
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

