; This testcase tests for a problem where LICM hoists 
; potentially trapping instructions when they are not guaranteed to execute.
;
; RUN: as < %s | opt -licm | dis | grep -C 2 "IfUnEqual" | grep div 

%X = global int 0
declare void %foo()

int %test(bool %c) {
	%A = load int *%X
	br label %Loop
Loop:
	call void %foo()
        br bool %c, label %LoopTail, label %IfUnEqual

IfUnEqual:
	%B1 = div int 4, %A  ;; Should not hoist this div!
	br label %LoopTail

LoopTail:
        %B = phi int [ 0, %Loop ], [ %B1, %IfUnEqual] 
        br bool %c, label %Loop, label %Out

Out:
	%C = sub int %A, %B
	ret int %C
}
