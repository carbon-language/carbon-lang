; RUN: llvm-as < %s | llc

	%struct..0anon = type { int }
	%struct.rtx_def = type { ushort, ubyte, ubyte, [1 x %struct..0anon] }

implementation   ; Functions:

fastcc void %immed_double_const(int %i0, int %i1) {
entry:
	%tmp1 = load uint* null		; <uint> [#uses=1]
	switch uint %tmp1, label %bb103 [
		 uint 1, label %bb
		 uint 3, label %bb
	]

bb:		; preds = %entry, %entry
	%tmp14 = setgt int 0, 31		; <bool> [#uses=1]
	br bool %tmp14, label %cond_next77, label %cond_next17

cond_next17:		; preds = %bb
	ret void

cond_next77:		; preds = %bb
	%tmp79.not = setne int %i1, 0		; <bool> [#uses=1]
	%tmp84 = setlt int %i0, 0		; <bool> [#uses=2]
	%bothcond1 = or bool %tmp79.not, %tmp84		; <bool> [#uses=1]
	br bool %bothcond1, label %bb88, label %bb99

bb88:		; preds = %cond_next77
	%bothcond2 = and bool false, %tmp84		; <bool> [#uses=0]
	ret void

bb99:		; preds = %cond_next77
	ret void

bb103:		; preds = %entry
	ret void
}
