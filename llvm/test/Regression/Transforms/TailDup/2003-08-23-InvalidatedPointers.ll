; RUN: as < %s | opt -tailduplicate -disable-output

implementation

int %sell_haggle() {
entry:		; No predecessors!
	br bool false, label %then.5, label %UnifiedExitNode

then.5:		; preds = %entry
	br bool false, label %loopentry.1.preheader, label %else.1

else.1:		; preds = %then.5
	br label %loopentry.1.preheader

loopentry.1.preheader:		; preds = %then.5, %else.1
	%final_ask.0 = phi int [ 0, %else.1 ], [ 0, %then.5 ]		; <int> [#uses=2]
	br label %loopentry.1

loopentry.1:		; preds = %loopentry.1.preheader, %endif.17
	switch uint 0, label %UnifiedExitNode [
		 uint 2, label %UnifiedExitNode
		 uint 1, label %endif.16
	]

endif.16:		; preds = %loopentry.1
	br bool false, label %then.17, label %UnifiedExitNode

then.17:		; preds = %endif.16
	br bool false, label %then.18, label %endif.17

then.18:		; preds = %then.17
	br bool false, label %endif.17, label %UnifiedExitNode

endif.17:		; preds = %then.17, %then.18
	%cur_ask.3 = phi int [ %final_ask.0, %then.17 ], [ %final_ask.0, %then.18 ]		; <int> [#uses=0]
	br bool false, label %loopentry.1, label %UnifiedExitNode

UnifiedExitNode:		; preds = %entry, %endif.17, %then.18, %endif.16, %loopentry.1, %loopentry.1
	ret int 0
}
