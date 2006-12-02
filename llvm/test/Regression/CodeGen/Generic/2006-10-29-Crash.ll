; RUN: llvm-upgrade < %s | llvm-as | llc

void %form_component_prediction(int %dy) {
entry:
	%tmp7 = and int %dy, 1		; <int> [#uses=1]
	%tmp27 = seteq int %tmp7, 0		; <bool> [#uses=1]
	br bool false, label %cond_next30, label %bb115

cond_next30:		; preds = %entry
	ret void

bb115:		; preds = %entry
	%bothcond1 = or bool %tmp27, false		; <bool> [#uses=1]
	br bool %bothcond1, label %bb228, label %cond_next125

cond_next125:		; preds = %bb115
	ret void

bb228:		; preds = %bb115
	ret void
}
