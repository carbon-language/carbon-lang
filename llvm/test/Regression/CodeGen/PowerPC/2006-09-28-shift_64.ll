; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64
target endian = big
target pointersize = 64
target triple = "powerpc64-apple-darwin8"

implementation   ; Functions:

void %glArrayElement_CompExec() {
entry:
	%tmp3 = and ulong 0, 18446744073701163007		; <ulong> [#uses=1]
	br label %cond_true24

cond_false:		; preds = %cond_true24
	ret void

cond_true24:		; preds = %cond_true24, %entry
	%indvar.ph = phi uint [ 0, %entry ], [ %indvar.next, %cond_true24 ]		; <uint> [#uses=1]
	%indvar = add uint 0, %indvar.ph		; <uint> [#uses=2]
	%code.0 = cast uint %indvar to ubyte		; <ubyte> [#uses=1]
	%tmp5 = add ubyte %code.0, 16		; <ubyte> [#uses=1]
	%tmp7 = shr ulong %tmp3, ubyte %tmp5		; <ulong> [#uses=1]
	%tmp7 = cast ulong %tmp7 to int		; <int> [#uses=1]
	%tmp8 = and int %tmp7, 1		; <int> [#uses=1]
	%tmp8 = seteq int %tmp8, 0		; <bool> [#uses=1]
	%indvar.next = add uint %indvar, 1		; <uint> [#uses=1]
	br bool %tmp8, label %cond_false, label %cond_true24
}
