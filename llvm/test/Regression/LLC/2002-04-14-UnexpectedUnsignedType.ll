; This causes the backend to assert out with:
; SparcInstrInfo.cpp:103: failed assertion `0 && "Unexpected unsigned type"'
;
implementation

declare void "bar"(sbyte* %G)

void "foo"()
begin
	%cast225 = cast ulong 123456 to sbyte*		; <sbyte*> [#uses=1]
	call void %bar( sbyte* %cast225)
	ret void
end
