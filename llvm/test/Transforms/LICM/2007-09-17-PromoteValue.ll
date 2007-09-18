; ModuleID = 'PR1657.bc'
; Do not promote getelementptr because it may exposes load from a null pointer 
; and store from a null pointer  which are covered by 
; icmp eq %struct.decision* null, null condition.
; RUN: llvm-as < %s | opt -licm | llvm-dis | not grep promoted
	%struct.decision = type { i8, %struct.decision* }

define i32 @main() {
entry:
	br label %blah.i

blah.i:		; preds = %cond_true.i, %entry
	%tmp3.i = icmp eq %struct.decision* null, null		; <i1> [#uses=1]
	br i1 %tmp3.i, label %clear_modes.exit, label %cond_true.i

cond_true.i:		; preds = %blah.i
	%tmp1.i = getelementptr %struct.decision* null, i32 0, i32 0		; <i8*> [#uses=1]
	store i8 0, i8* %tmp1.i
	br label %blah.i

clear_modes.exit:		; preds = %blah.i
	call void @exit( i32 0 )
	unreachable
}

declare void @exit(i32)
