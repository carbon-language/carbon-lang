; Assertion failure in LevelRaise caused by this code:
;
; LevelRaise.cpp:92: failed assertion `I->getOpcode() == Instruction::Add && I->getNumOperands() == 2 && "Use is not a valid add instruction!"'
;
; RUN: as < %s | opt -raise

%lit_base = uninitialized global [26 x int]		; <[26 x int] *> [#uses=1]
%leaves = uninitialized global [26 x int]		; <[26 x int] *> [#uses=1]
%parents = uninitialized global [26 x int]		; <[26 x int] *> [#uses=1]
implementation

int "build_tree"(int %ml)
begin
; <label>:0					;[#uses=2]
	%ml = alloca int		; <int *> [#uses=2]
	store int %ml, int * %ml
	%reg107 = load int * %ml		; <int> [#uses=1]
	br label %bb2

bb2:					;[#uses=4]
	%reg137 = phi int [ %reg140, %bb2 ], [ %reg107, %0 ]		; <int> [#uses=2]
	%reg138 = phi uint [ %reg139, %bb2 ], [ 0, %0 ]		; <uint> [#uses=3]
	%cast1005 = cast int %reg137 to uint		; <uint> [#uses=1]
	%reg111 = shl uint %cast1005, ubyte 2		; <uint> [#uses=1]
	%cast112 = cast uint %reg111 to sbyte *		; <sbyte *> [#uses=3]
	%cast1002 = cast [26 x int] * %parents to sbyte *		; <sbyte *> [#uses=1]
	%reg115 = add sbyte * %cast112, %cast1002		; <sbyte *> [#uses=1]
	;%cast1006 = cast sbyte * %reg115 to uint *		; <uint *> [#uses=1]

	%cast1003 = cast [26 x int] * %lit_base to sbyte *		; <sbyte *> [#uses=1]
	%reg121 = add sbyte * %cast112, %cast1003		; <sbyte *> [#uses=2]
	%cast1007 = cast sbyte * %reg121 to uint *		; <uint *> [#uses=1]
	%reg128 = load uint * %cast1007		; <uint> [#uses=1]
	%reg129 = sub uint %reg128, %reg138		; <uint> [#uses=1]
	%cast1008 = cast sbyte * %reg121 to uint *		; <uint *> [#uses=1]
	;store uint %reg129, uint * %cast1008
	%cast1004 = cast [26 x int] * %leaves to sbyte *		; <sbyte *> [#uses=1]
	%reg135 = add sbyte * %cast112, %cast1004		; <sbyte *> [#uses=1]
	%cast1009 = cast sbyte * %reg135 to uint *		; <uint *> [#uses=1]
	%reg136 = load uint * %cast1009		; <uint> [#uses=1]
	%reg139 = add uint %reg138, %reg136		; <uint> [#uses=1]
	%reg140 = add int %reg137, -1		; <int> [#uses=1]
	br bool false, label %bb2, label %bb3

bb3:					;[#uses=1]
	ret int %reg137
end
