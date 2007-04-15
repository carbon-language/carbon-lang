; This testcase checks to see if the simplifycfg pass is converting invoke
; instructions to call instructions if the handler just rethrows the exception.

; If this test is successful, the function should be reduced to 'call; ret'

; RUN: llvm-upgrade < %s | llvm-as | opt -simplifycfg | llvm-dis | \
; RUN:   not egrep {\\(invoke\\)|\\(br\\)}

declare void %bar()

int %test() {
	invoke void %bar() to label %Ok except label %Rethrow
Ok:
	ret int 0
Rethrow:
	unwind
}

