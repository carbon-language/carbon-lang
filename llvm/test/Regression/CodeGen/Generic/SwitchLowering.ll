; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 | grep cmp | wc -l | grep 1
; PR964

sbyte* %FindChar(sbyte* %CurPtr) {
entry:
	br label %bb

bb:		; preds = %bb, %entry
	%indvar = phi uint [ 0, %entry ], [ %indvar.next, %bb ]		; <uint> [#uses=3]
	%CurPtr_addr.0.rec = cast uint %indvar to int		; <int> [#uses=1]
	%CurPtr_addr.0 = getelementptr sbyte* %CurPtr, uint %indvar		; <sbyte*> [#uses=1]
	%tmp = load sbyte* %CurPtr_addr.0		; <sbyte> [#uses=2]
	%tmp2.rec = add int %CurPtr_addr.0.rec, 1		; <int> [#uses=1]
	%tmp2 = getelementptr sbyte* %CurPtr, int %tmp2.rec		; <sbyte*> [#uses=1]
	%indvar.next = add uint %indvar, 1		; <uint> [#uses=1]
	switch sbyte %tmp, label %bb [
		 sbyte 0, label %bb7
		 sbyte 120, label %bb7
	]

bb7:		; preds = %bb, %bb
	%tmp = cast sbyte %tmp to ubyte		; <ubyte> [#uses=1]
	tail call void %foo( ubyte %tmp )
	ret sbyte* %tmp2
}

declare void %foo(ubyte)
