; RUN: llvm-as < %s | llc -march=x86 -stats 2>&1 | grep "asm-printer" | grep 19
; XFAIL: *

void %_ZN9__gnu_cxx9hashtableISt4pairIKPKciES3_NS_4hashIS3_EESt10_Select1stIS5_E5eqstrSaIiEE14find_or_insertERKS5__cond_true456.i(sbyte* %tmp435.i, uint* %tmp449.i.out) {
newFuncRoot:
	br label %cond_true456.i

bb459.i.exitStub:		; preds = %cond_true456.i
	store uint %tmp449.i, uint* %tmp449.i.out
	ret void

cond_true456.i:		; preds = %cond_true456.i, %newFuncRoot
	%__s441.2.4.i = phi sbyte* [ %tmp451.i, %cond_true456.i ], [ %tmp435.i, %newFuncRoot ]		; <sbyte*> [#uses=2]
	%__h.2.4.i = phi uint [ %tmp449.i, %cond_true456.i ], [ 0, %newFuncRoot ]		; <uint> [#uses=1]
	%tmp446.i = mul uint %__h.2.4.i, 5		; <uint> [#uses=1]
	%tmp.i = load sbyte* %__s441.2.4.i		; <sbyte> [#uses=1]
	%tmp448.i = cast sbyte %tmp.i to uint		; <uint> [#uses=1]
	%tmp449.i = add uint %tmp448.i, %tmp446.i		; <uint> [#uses=2]
	%tmp450.i = cast sbyte* %__s441.2.4.i to uint		; <uint> [#uses=1]
	%tmp451.i = add uint %tmp450.i, 1		; <uint> [#uses=1]
	%tmp451.i = cast uint %tmp451.i to sbyte*		; <sbyte*> [#uses=2]
	%tmp45435.i = load sbyte* %tmp451.i		; <sbyte> [#uses=1]
	%tmp45536.i = seteq sbyte %tmp45435.i, 0		; <bool> [#uses=1]
	br bool %tmp45536.i, label %bb459.i.exitStub, label %cond_true456.i
}
