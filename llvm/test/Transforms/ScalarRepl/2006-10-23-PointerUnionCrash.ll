; RUN: llvm-upgrade < %s | llvm-as | opt -scalarrepl -disable-output

target datalayout = "e-p:32:32"
target endian = little
target pointersize = 32
target triple = "i686-apple-darwin8.7.2"

implementation   ; Functions:

void %glgProcessColor() {
entry:
	%source_ptr = alloca sbyte*, align 4		; <sbyte**> [#uses=2]
	br bool false, label %bb1357, label %cond_next583

cond_next583:		; preds = %entry
	ret void

bb1357:		; preds = %entry
	br bool false, label %bb1365, label %bb27055

bb1365:		; preds = %bb1357
	switch uint 0, label %cond_next10377 [
		 uint 0, label %bb4679
		 uint 1, label %bb4679
		 uint 2, label %bb4679
		 uint 3, label %bb4679
		 uint 4, label %bb5115
		 uint 5, label %bb6651
		 uint 6, label %bb7147
		 uint 7, label %bb8683
		 uint 8, label %bb9131
		 uint 9, label %bb9875
		 uint 10, label %bb4679
		 uint 11, label %bb4859
		 uint 12, label %bb4679
		 uint 16, label %bb10249
	]

bb4679:		; preds = %bb1365, %bb1365, %bb1365, %bb1365, %bb1365, %bb1365
	ret void

bb4859:		; preds = %bb1365
	ret void

bb5115:		; preds = %bb1365
	ret void

bb6651:		; preds = %bb1365
	ret void

bb7147:		; preds = %bb1365
	ret void

bb8683:		; preds = %bb1365
	ret void

bb9131:		; preds = %bb1365
	ret void

bb9875:		; preds = %bb1365
	%source_ptr9884 = cast sbyte** %source_ptr to ubyte**		; <ubyte**> [#uses=1]
	%tmp9885 = load ubyte** %source_ptr9884		; <ubyte*> [#uses=0]
	ret void

bb10249:		; preds = %bb1365
	%source_ptr10257 = cast sbyte** %source_ptr to ushort**		; <ushort**> [#uses=1]
	%tmp10258 = load ushort** %source_ptr10257		; <ushort*> [#uses=0]
	ret void

cond_next10377:		; preds = %bb1365
	ret void

bb27055:		; preds = %bb1357
	ret void
}
