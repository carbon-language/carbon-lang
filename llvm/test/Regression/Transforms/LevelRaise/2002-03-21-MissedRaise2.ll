; Problem that occured because of obsolete code that only allowed malloc
; instructions to change type if they were 'array' allocations.  This
; prevented reg115 from being able to change.
;

; RUN: llvm-as < %s | opt -raise | llvm-dis | grep '= cast' | not grep \*
	
	%Hash = type { { uint, sbyte *, \2 } * *, int (uint) *, int } *
	%HashEntry = type { uint, sbyte *, \2 } *
	%hash = type { { uint, sbyte *, \2 } * *, int (uint) *, int }
	%hash_entry = type { uint, sbyte *, \2 * }
implementation

%Hash "MakeHash"(int %size, int (uint) * %map)
begin
bb1:					;[#uses=0]
	%reg112 = malloc sbyte * *, uint 3		; <sbyte * * *> [#uses=4]
	%reg115 = malloc sbyte *, uint 1		; <sbyte * *> [#uses=1]
	store sbyte * * %reg115, sbyte * * * %reg112
	%reg121 = load sbyte * * * %reg112		; <sbyte * *> [#uses=1]
	%size-idxcast1 = cast int %size to long		; <uint> [#uses=1]
	%reg1221 = getelementptr sbyte * * %reg121, long %size-idxcast1		; <sbyte * *> [#uses=1]
	store sbyte * null, sbyte * * %reg1221
	%reg232 = getelementptr sbyte * * * %reg112, long 1		; <sbyte * * *> [#uses=1]
	%cast243 = cast int (uint) * %map to sbyte * *		; <sbyte * *> [#uses=1]
	store sbyte * * %cast243, sbyte * * * %reg232
	%cast246 = cast sbyte * * * %reg112 to %Hash		; <%Hash> [#uses=1]
	ret %Hash %cast246
end
