; Test that bugpoint can narrow down the testcase to the important function
;
; RUN: bugpoint %s -bugpoint-crashcalls

int %foo() { ret int 1 }

int %test() {
	call int %test()
	ret int %0
}

int %bar() { ret int 2 }

