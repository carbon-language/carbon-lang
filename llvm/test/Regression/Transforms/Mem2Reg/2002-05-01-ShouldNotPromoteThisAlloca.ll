; This input caused the mem2reg pass to die because it was trying to promote
; the %r alloca, even though it is invalid to do so in this case!
;
; RUN: llvm-as < %s | opt -mem2reg


implementation

void "test"()
begin
	%r = alloca int		; <int*> [#uses=2]
	store int 4, int* %r
	store int* %r, int** null
	ret void
end
