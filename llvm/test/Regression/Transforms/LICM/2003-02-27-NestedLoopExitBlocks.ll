; Exit blocks need to be updated for all nested loops...

; RUN: as < %s | opt -preheaders

implementation   ; Functions:

int %yyparse() {
bb0:		; No predecessors!
	br bool false, label %UnifiedExitNode, label %bb19

bb19:		; preds = %bb28, %bb0
	br bool false, label %bb28, label %UnifiedExitNode

bb28:		; preds = %bb32, %bb19
	br bool false, label %bb32, label %bb19

bb32:		; preds = %bb28
	br bool false, label %UnifiedExitNode, label %bb28

UnifiedExitNode:		; preds = %bb32, %bb19, %bb0
	ret int 0
}
