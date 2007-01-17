; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm

%handler_installed.6144.b = external global bool                ; <bool*> [#uses=1]


void %__mf_sigusr1_respond() {
entry:
	%tmp8.b = load bool* %handler_installed.6144.b		; <bool> [#uses=1]
	br bool false, label %cond_true7, label %cond_next

cond_next:		; preds = %entry
	br bool %tmp8.b, label %bb, label %cond_next3

cond_next3:		; preds = %cond_next
	ret void

bb:		; preds = %cond_next
	ret void

cond_true7:		; preds = %entry
	ret void
}
