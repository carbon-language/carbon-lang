; Test that bugpoint can narrow down the testcase to the important function
;
; RUN: llvm-upgrade < %s > %t1.ll
; RUN: bugpoint %t1.ll -bugpoint-crashcalls

int %foo() { ret int 1 }

int %test() {
	call int %test()
	ret int %0
}

int %bar() { ret int 2 }

