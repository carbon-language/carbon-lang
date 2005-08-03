; RUN: llvm-as < %s | opt -lowerinvoke -enable-correct-eh-support -disable-output

declare fastcc int %ll_listnext__listiterPtr()

fastcc int %WorkTask.fn() {
block0:
	%v2679 = invoke fastcc int %ll_listnext__listiterPtr( )
			to label %block9 unwind label %block8_exception_handling		; <int> [#uses=1]

block8_exception_handling:		; preds = %block0
	ret int 0

block9:		; preds = %block0
	%i_2689 = phi int [ %v2679, %block0 ]		; <int> [#uses=0]
	ret int %i_2689
}
