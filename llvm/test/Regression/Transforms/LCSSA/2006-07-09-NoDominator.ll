; RUN: llvm-as < %s | opt -lcssa

	%struct.SetJmpMapEntry = type { sbyte*, uint, %struct.SetJmpMapEntry* }

implementation   ; Functions:

void %__llvm_sjljeh_try_catching_longjmp_exception() {
entry:
	br label %loopentry

loopentry:		; preds = %endif, %entry
	%SJE.0 = phi %struct.SetJmpMapEntry* [ null, %entry ], [ %tmp.25, %endif ]		; <%struct.SetJmpMapEntry*> [#uses=1]
	br bool false, label %no_exit, label %loopexit

no_exit:		; preds = %loopentry
	br bool false, label %then, label %endif

then:		; preds = %no_exit
	%tmp.21 = getelementptr %struct.SetJmpMapEntry* %SJE.0, int 0, uint 1		; <uint*> [#uses=0]
	br label %return

endif:		; preds = %after_ret.0, %no_exit
	%tmp.25 = load %struct.SetJmpMapEntry** null		; <%struct.SetJmpMapEntry*> [#uses=1]
	br label %loopentry

loopexit:		; preds = %loopentry
	br label %return

return:		; preds = %after_ret.1, %loopexit, %then
	ret void
}
