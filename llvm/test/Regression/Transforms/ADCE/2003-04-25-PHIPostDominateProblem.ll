; THis testcase caused an assertion failure because a PHI node did not have 
; entries for it's postdominator.  But I think this can only happen when the 
; PHI node is dead, so we just avoid patching up dead PHI nodes.

; RUN: llvm-as < %s | opt -adce

target endian = little
target pointersize = 32

implementation   ; Functions:

void %dead_test8() {
entry:		; No predecessors!
	br label %loopentry

loopentry:		; preds = %entry, %endif
	%k.1 = phi int [ %k.0, %endif ], [ 0, %entry ]		; <int> [#uses=1]
	br bool false, label %no_exit, label %return

no_exit:		; preds = %loopentry
	br bool false, label %then, label %else

then:		; preds = %no_exit
	br label %endif

else:		; preds = %no_exit
	%dec = add int %k.1, -1		; <int> [#uses=1]
	br label %endif

endif:		; preds = %else, %then
	%k.0 = phi int [ %dec, %else ], [ 0, %then ]		; <int> [#uses=1]
	store int 2, int* null
	br label %loopentry

return:		; preds = %loopentry
	ret void
}
