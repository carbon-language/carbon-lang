; RUN: analyze -exprs %s

implementation

void "foo"(int %reg126)
begin
	%cast1007 = cast int %reg126 to uint		; <uint> [#uses=1]

	%reg119 = sub uint %cast1007, %cast1007		; <uint> [#uses=1]
	%cast121 = cast uint %reg119 to sbyte *		; <sbyte *> [#uses=1]

	ret void
end
