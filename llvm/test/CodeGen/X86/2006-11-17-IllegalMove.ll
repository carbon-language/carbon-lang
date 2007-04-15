; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86-64 | \
; RUN:   not grep {movb %sil, %ah}
; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86-64 | \
; RUN:   grep {movzbw %al, %ax}

void %handle_vector_size_attribute() {
entry:
	%tmp69 = load uint* null		; <uint> [#uses=1]
	switch uint %tmp69, label %bb84 [
		 uint 2, label %bb77
		 uint 1, label %bb77
	]

bb77:		; preds = %entry, %entry
	%tmp99 = udiv ulong 0, 0		; <ulong> [#uses=1]
	%tmp = load ubyte* null		; <ubyte> [#uses=1]
	%tmp114 = seteq ulong 0, 0		; <bool> [#uses=1]
	br bool %tmp114, label %cond_true115, label %cond_next136

bb84:		; preds = %entry
	ret void

cond_true115:		; preds = %bb77
	%tmp118 = load ubyte* null		; <ubyte> [#uses=1]
	br bool false, label %cond_next129, label %cond_true120

cond_true120:		; preds = %cond_true115
	%tmp127 = udiv ubyte %tmp, %tmp118		; <ubyte> [#uses=1]
	%tmp127 = cast ubyte %tmp127 to ulong		; <ulong> [#uses=1]
	br label %cond_next129

cond_next129:		; preds = %cond_true120, %cond_true115
	%iftmp.30.0 = phi ulong [ %tmp127, %cond_true120 ], [ 0, %cond_true115 ]		; <ulong> [#uses=1]
	%tmp132 = seteq ulong %iftmp.30.0, %tmp99		; <bool> [#uses=1]
	br bool %tmp132, label %cond_false148, label %cond_next136

cond_next136:		; preds = %cond_next129, %bb77
	ret void

cond_false148:		; preds = %cond_next129
	ret void
}
