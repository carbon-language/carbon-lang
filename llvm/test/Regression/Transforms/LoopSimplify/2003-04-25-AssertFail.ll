; This testcase exposed a problem with the loop identification pass (LoopInfo).
; Basically, it was incorrectly calculating the loop nesting information.
;
; RUN: llvm-as < %s | opt -preheaders

implementation   ; Functions:

int %yylex() {		; No predecessors!
	br label %loopentry.0

loopentry.0:		; preds = %0, %yy_find_action, %else.4
	br label %loopexit.2

loopexit.2:		; preds = %loopentry.0, %else.4, %loopexit.2
	br bool false, label %loopexit.2, label %else.4

yy_find_action:		; preds = %loopexit.2, %else.4
	br label %else.4

else.4:		; preds = %yy_find_action
	switch uint 0, label %loopexit.2 [
		 uint 2, label %yy_find_action
		 uint 0, label %loopentry.0
	]
}
