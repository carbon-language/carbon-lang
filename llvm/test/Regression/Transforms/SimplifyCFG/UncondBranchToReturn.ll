; The unify-function-exit-nodes pass often makes basic blocks that just contain
; a PHI node and a return.  Make sure the simplify cfg can straighten out this
; important case.  This is basically the most trivial form of tail-duplication.

; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | not grep 'br label'

int %test(bool %B, int %A, int %B) {
	br bool %B, label %T, label %F
T:
	br label %ret
F:
	br label %ret
ret:
	%X = phi int [%A, %F], [%B, %T]
	ret int %X
}

; Make sure it's willing to move unconditional branches to return instructions
; as well, even if the return block is shared and the source blocks are
; non-empty.
int %test2(bool %B, int %A, int %B) {
	br bool %B, label %T, label %F
T:
	call int %test(bool true, int 5, int 8)
	br label %ret
F:
	call int %test(bool true, int 5, int 8)
	br label %ret
ret:
	ret int %A
}

