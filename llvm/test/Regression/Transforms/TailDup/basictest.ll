; RUN: as < %s | opt -tailduplicate -disable-output

declare void %__main()

int %main() {
entry:		; No predecessors!
	call void %__main( )
	br label %loopentry

loopentry:		; preds = %entry, %no_exit
	%i.0 = phi int [ %inc, %no_exit ], [ 0, %entry ]		; <int> [#uses=2]
	%tmp.1 = setle int %i.0, 99		; <bool> [#uses=1]
	br bool %tmp.1, label %no_exit, label %return

no_exit:		; preds = %loopentry
	%tmp.51 = call int %main( )		; <int> [#uses=0]
	%inc = add int %i.0, 1		; <int> [#uses=1]
	br label %loopentry

return:		; preds = %loopentry
	ret int %i.0
}
