; RUN: llvm-as < %s | opt -scalarrepl -disable-output

void %output_toc() {
entry:
	%buf = alloca [256 x sbyte], align 16		; <[256 x sbyte]*> [#uses=1]
	%name = alloca sbyte*, align 4		; <sbyte**> [#uses=1]
	%real_name = alloca sbyte*, align 4		; <sbyte**> [#uses=0]
	"alloca point" = cast int 0 to int		; <int> [#uses=0]
	%buf = cast [256 x sbyte]* %buf to sbyte*		; <sbyte*> [#uses=1]
	store sbyte* %buf, sbyte** %name
	call void %abort( )
	unreachable

return:		; No predecessors!
	ret void
}

declare void %abort()
