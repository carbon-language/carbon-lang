; RUN: llvm-as < %s | opt -cee
;
; This testcase causes an assertion error.
;
implementation   ; Functions:

void %test(int %A) {
	br label %bb2
bb2:
	ret void

bb3:		; No predecessors!
	br bool true, label %bb4, label %bb2

bb4:
	ret void
}
