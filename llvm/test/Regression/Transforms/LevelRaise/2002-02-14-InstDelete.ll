; The -raise pass is not correctly deleting instructions in all cases where it
; should, which leaves the instruction to be deleted by DCE.  This is bad
; because some instructions are left in an invalid form, causing an assertion
; failure.  In this case, a chain of instructions, including a zero argument;
; PHI node is left behind.
;
; RUN: as < %s | opt -raise -q
;
	%Hash = type { { uint, sbyte *, \2 } * *, int (uint) *, int } *
	%HashEntry = type { uint, sbyte *, \2 } *
	%hash = type { { uint, sbyte *, \2 } * *, int (uint) *, int }
	%hash_entry = type { uint, sbyte *, \2 * }
implementation

void "HashDelete"(uint %key, %Hash %hash)
begin
bb1:					;[#uses=1]
	%cast1006 = cast %Hash %hash to sbyte *		; <sbyte *> [#uses=1]
	%cast1007 = cast ulong 8 to sbyte *		; <sbyte *> [#uses=1]
	%reg1000 = add sbyte * %cast1006, %cast1007		; <sbyte *> [#uses=1]
	%cast1008 = cast sbyte * %reg1000 to sbyte * *		; <sbyte * *> [#uses=1]
	%reg111 = load sbyte * * %cast1008		; <sbyte *> [#uses=1]
	%cast1009 = cast sbyte * %reg111 to uint (...) *		; <uint (...) *> [#uses=1]
	%reg110 = call uint (...) * %cast1009( uint %key )		; <uint> [#uses=1]
	%reg114 = shl uint %reg110, ubyte 3		; <uint> [#uses=1]
	%cast115 = cast uint %reg114 to ulong		; <ulong> [#uses=1]
	%cast116 = cast ulong %cast115 to sbyte *		; <sbyte *> [#uses=1]
	%cast1010 = cast %Hash %hash to sbyte * *		; <sbyte * *> [#uses=1]
	%reg117 = load sbyte * * %cast1010		; <sbyte *> [#uses=1]
	%reg112 = add sbyte * %reg117, %cast116		; <sbyte *> [#uses=4]
	%cast1011 = cast sbyte * %reg112 to sbyte * *		; <sbyte * *> [#uses=1]
	%reg122 = load sbyte * * %cast1011		; <sbyte *> [#uses=2]
	%cast1012 = cast ulong 0 to sbyte *		; <sbyte *> [#uses=1]
	%cond1001 = seteq sbyte * %reg122, %cast1012		; <bool> [#uses=1]
	br bool %cond1001, label %bb5, label %bb2

bb2:					;[#uses=3]
	%cast1013 = cast sbyte * %reg122 to uint *		; <uint *> [#uses=1]
	%reg124 = load uint * %cast1013		; <uint> [#uses=1]
	%cond1002 = seteq uint %reg124, %key		; <bool> [#uses=1]
	br bool %cond1002, label %bb5, label %bb3

bb3:					;[#uses=3]
	%reg125 = phi sbyte * [ %reg126, %bb4 ], [ %reg112, %bb2 ]		; <sbyte *> [#uses=1]
	%cast1014 = cast sbyte * %reg125 to sbyte * *		; <sbyte * *> [#uses=1]
	%reg121 = load sbyte * * %cast1014		; <sbyte *> [#uses=2]
	%cast1015 = cast ulong 16 to sbyte *		; <sbyte *> [#uses=1]
	%reg126 = add sbyte * %reg121, %cast1015		; <sbyte *> [#uses=3]
	%cast1016 = cast ulong 16 to sbyte *		; <sbyte *> [#uses=1]
	%reg1003 = add sbyte * %reg121, %cast1016		; <sbyte *> [#uses=1]
	%cast1017 = cast sbyte * %reg1003 to sbyte * *		; <sbyte * *> [#uses=1]
	%reg118 = load sbyte * * %cast1017		; <sbyte *> [#uses=2]
	%cast1018 = cast ulong 0 to sbyte *		; <sbyte *> [#uses=1]
	%cond1004 = seteq sbyte * %reg118, %cast1018		; <bool> [#uses=1]
	br bool %cond1004, label %bb5, label %bb4

bb4:					;[#uses=3]
	%cast1019 = cast sbyte * %reg118 to uint *		; <uint *> [#uses=1]
	%reg120 = load uint * %cast1019		; <uint> [#uses=1]
	%cond1005 = setne uint %reg120, %key		; <bool> [#uses=1]
	br bool %cond1005, label %bb3, label %bb5

bb5:					;[#uses=4]
	%reg127 = phi sbyte * [ %reg126, %bb4 ], [ %reg126, %bb3 ], [ %reg112, %bb2 ], [ %reg112, %bb1 ]		; <sbyte *> [#uses=1]
	call int (...) * %foo( sbyte * %reg127 )		; <int>:0 [#uses=0]
	ret void
end

declare int "foo"(...)
