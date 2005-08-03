; RUN: llvm-as < %s | opt -lowerinvoke -enable-correct-eh-support -disable-output

implementation   ; Functions:

declare void %ll_listnext__listiterPtr()

void %WorkTask.fn() {
block0:
	invoke void %ll_listnext__listiterPtr( )
			to label %block9 unwind label %block8_exception_handling

block8_exception_handling:		; preds = %block0
	ret void

block9:		; preds = %block0
	%w_2690 = phi { int, int }* [ null, %block0 ]		; <{ int, int }*> [#uses=1]
	%tmp.129 = getelementptr { int, int }* %w_2690, int 0, uint 1		; <int*> [#uses=1]
	%v2769 = load int* %tmp.129		; <int> [#uses=0]
	ret void
}
