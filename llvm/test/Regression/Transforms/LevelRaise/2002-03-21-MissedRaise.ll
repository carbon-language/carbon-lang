; This example should be raised to return a Hash directly without casting.  To
; successful, all cast instructions should be eliminated from this testcase.
;

; RUN: as < %s | opt -raise | dis | not grep cast 

	%Hash = type { { uint, sbyte *, \2 } * *, int (uint) *, int } *
	%hash = type { { uint, sbyte *, \2 } * *, int (uint) *, int }
	%hash_el = type { uint, sbyte *, \2 } *
implementation

%Hash "MakeHash"(int %size, int (uint) * %map)
begin
	%reg112 = malloc sbyte, uint 24		; <sbyte *> [#uses=5]
	%reg115 = malloc sbyte, uint 96		; <sbyte *> [#uses=1]
	%cast237 = cast sbyte * %reg112 to sbyte * *		; <sbyte * *> [#uses=1]
	store sbyte * %reg115, sbyte * * %cast237

	%cast246 = cast sbyte * %reg112 to %Hash		; <%Hash> [#uses=1]
	ret %Hash %cast246
end

