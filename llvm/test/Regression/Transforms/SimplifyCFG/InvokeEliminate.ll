; This testcase checks to see if the simplifycfg pass is converting invoke
; instructions to call instructions if the handler just rethrows the exception.

; If this test is successful, the function should be reduced to 'call; ret'

; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | not egrep 'invoke|br'

declare void %bar()
declare void %llvm.unwind()

int %test() {
	invoke void %bar() to label %Ok except label %Rethrow
Ok:
	ret int 0
Rethrow:
	call void %llvm.unwind()
	br label %Ok
}

