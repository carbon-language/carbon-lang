; RUN: llvm-as < %s | opt -basicaa -licm -disable-output

;%MoveArray = external global [64 x ulong]

implementation   ; Functions:

void %InitMoveArray() {
bb3:		; No predecessors!
	%X = alloca [2 x ulong]
	br bool false, label %bb13, label %bb4

bb4:		; preds = %bb3
	%reg3011 = getelementptr [2 x ulong]* %X, long 0, long 0
	br label %bb8

bb8:		; preds = %bb8, %bb4
	store ulong 0, ulong* %reg3011
	br bool false, label %bb8, label %bb13

bb13:		; preds = %bb8, %bb3
	ret void
}
