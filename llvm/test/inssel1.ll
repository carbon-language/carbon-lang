implementation

void "foo"(int * %x, int * %y)
begin
; <label>:0					;[#uses=0]
	seteq int * %x, %y		; <bool>:0	[#uses=0]
	ret void
end
