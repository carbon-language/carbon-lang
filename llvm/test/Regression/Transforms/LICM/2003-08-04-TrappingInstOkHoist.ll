; This testcase tests to make sure a trapping instruction is hoisted when
; it is guaranteed to execute.
;
; RUN: llvm-as < %s | opt -licm | llvm-dis | grep -C 2 "test" | grep div

%X = global int 0
declare void %foo()

int %test(bool %c) {
	%A = load int *%X
	br label %Loop
Loop:
	call void %foo()
	%B = div int 4, %A  ;; Should have hoisted this div!
        br bool %c, label %Loop, label %Out

Out:
	%C = sub int %A, %B
	ret int %C
}
