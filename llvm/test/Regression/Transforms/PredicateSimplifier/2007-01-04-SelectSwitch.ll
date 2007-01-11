; RUN: llvm-upgrade < %s | llvm-as | opt -predsimplify -disable-output

void %ercMarkCurrMBConcealed(int %comp) {
entry:
	%tmp5 = icmp slt int %comp, 0		; <bool> [#uses=2]
	%comp_addr.0 = select bool %tmp5, int 0, int %comp		; <int> [#uses=1]
	switch int %comp_addr.0, label %return [
		 int 0, label %bb
	]

bb:		; preds = %entry
	br bool %tmp5, label %bb87.bb97_crit_edge.critedge, label %return

bb87.bb97_crit_edge.critedge:		; preds = %bb
	ret void

return:		; preds = %bb, %entry
	ret void
}
