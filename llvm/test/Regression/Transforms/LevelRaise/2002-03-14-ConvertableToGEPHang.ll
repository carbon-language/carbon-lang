; This testcase found a bug in ConvertableToGEP that could cause an infinite loop
; Note that this code is actually miscompiled from the input source, but despite 
; that, level raise should not hang!
;
; RUN: llvm-as < %s | opt -raise	
	
	%Disjunct = type { \2 *, short, sbyte, sbyte *, { short, short, sbyte, sbyte, \2, sbyte * } *, { short, short, sbyte, sbyte, \2, sbyte * } * }
%chosen_disjuncts = uninitialized global %Disjunct * *		; <%Disjunct * * *> [#uses=1]
implementation

void "build_image_array"()
begin
bb0:					;[#uses=0]
	%reg109 = getelementptr %Disjunct * * * %chosen_disjuncts, long 7		; <%Disjunct * * *> [#uses=1]
	%reg108 = load %Disjunct * * * %reg109		; <%Disjunct * *> [#uses=1]
	%reg1000 = getelementptr %Disjunct * * %reg108, long 3		; <%Disjunct * *> [#uses=1]
	%cast1007 = cast %Disjunct * * %reg1000 to sbyte * *		; <sbyte * *> [#uses=1]
	%reg110 = load sbyte * * %cast1007		; <sbyte *> [#uses=1]
	%cast1008 = cast ulong 4 to sbyte *		; <sbyte *> [#uses=1]
	%A = cast sbyte * %reg110 to ulong
	%B = cast sbyte * %cast1008 to ulong
	%reg1001 = add ulong %A, %B		; <sbyte *> [#uses=0]
	ret void
end
