; This testcase was incorrectly computing that the loopentry.7 loop was
; not a child of the loopentry.6 loop.
;
; RUN: analyze %s -loops | grep "^        Loop Containing:  %loopentry.7"

void %getAndMoveToFrontDecode() {		; No predecessors!
	br label %endif.2

endif.2:		; preds = %0, %loopexit.5
	br bool false, label %loopentry.5, label %UnifiedExitNode

loopentry.5:		; preds = %endif.2, %loopexit.6
	br bool false, label %loopentry.6, label %UnifiedExitNode

loopentry.6:		; preds = %loopentry.5, %loopentry.7
	br bool false, label %loopentry.7, label %loopexit.6

loopentry.7:		; preds = %loopentry.6, %loopentry.7
	br bool false, label %loopentry.7, label %loopentry.6

loopexit.6:		; preds = %loopentry.6
	br bool false, label %loopentry.5, label %loopexit.5

loopexit.5:		; preds = %loopexit.6
	br bool false, label %endif.2, label %UnifiedExitNode

UnifiedExitNode:		; preds = %endif.2, %loopexit.5, %loopentry.5
	ret void
}
