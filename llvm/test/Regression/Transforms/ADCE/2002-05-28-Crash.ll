; This testcase is distilled from the GNU rx package.  The loop should be 
; removed but causes a problem when ADCE does.  The source function is:
; int rx_bitset_empty (int size, rx_Bitset set) {
;  int x;
;  RX_subset s;
;  s = set[0];
;  set[0] = 1;
;  for (x = rx_bitset_numb_subsets(size) - 1; !set[x]; --x)
;    ;
;  set[0] = s;
;  return !s;
;}
;
; RUN: llvm-as < %s | opt -adce

implementation   ; Functions:

int %rx_bitset_empty(int %size, uint* %set) {
bb1:					;[#uses=2]
	%reg110 = load uint* %set		; <uint> [#uses=2]
	store uint 1, uint* %set
	%cast112 = cast int %size to ulong		; <ulong> [#uses=1]
	%reg113 = add ulong %cast112, 31		; <ulong> [#uses=1]
	%reg114 = shr ulong %reg113, ubyte 5		; <ulong> [#uses=2]
	%cast109 = cast ulong %reg114 to int		; <int> [#uses=1]
	%reg129 = add int %cast109, -1		; <int> [#uses=1]
	%reg114-idxcast = cast ulong %reg114 to uint		; <uint> [#uses=1]
	%reg114-idxcast-offset = add uint %reg114-idxcast, 1073741823		; <uint> [#uses=1]
	%reg114-idxcast-offset = cast uint %reg114-idxcast-offset to long
	%reg124 = getelementptr uint* %set, long %reg114-idxcast-offset		; <uint*> [#uses=1]
	%reg125 = load uint* %reg124		; <uint> [#uses=1]
	%cond232 = setne uint %reg125, 0		; <bool> [#uses=1]
	br bool %cond232, label %bb3, label %bb2

bb2:					;[#uses=3]
	%cann-indvar = phi int [ 0, %bb1 ], [ %add1-indvar, %bb2 ]		; <int> [#uses=2]
	%reg130-scale = mul int %cann-indvar, -1		; <int> [#uses=1]
	%reg130 = add int %reg130-scale, %reg129		; <int> [#uses=1]
	%add1-indvar = add int %cann-indvar, 1		; <int> [#uses=1]
	%reg130-idxcast = cast int %reg130 to uint		; <uint> [#uses=1]
	%reg130-idxcast-offset = add uint %reg130-idxcast, 1073741823		; <uint> [#uses=1]
	%reg130-idxcast-offset = cast uint %reg130-idxcast-offset to long
	%reg118 = getelementptr uint* %set, long %reg130-idxcast-offset		; <uint*> [#uses=1]
	%reg119 = load uint* %reg118		; <uint> [#uses=1]
	%cond233 = seteq uint %reg119, 0		; <bool> [#uses=1]
	br bool %cond233, label %bb2, label %bb3

bb3:					;[#uses=2]
	store uint %reg110, uint* %set
	%cast126 = cast uint %reg110 to ulong		; <ulong> [#uses=1]
	%reg127 = add ulong %cast126, 18446744073709551615		; <ulong> [#uses=1]
	%reg128 = shr ulong %reg127, ubyte 63		; <ulong> [#uses=1]
	%cast120 = cast ulong %reg128 to int		; <int> [#uses=1]
	ret int %cast120

}
