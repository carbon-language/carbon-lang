; Verify the the operand type of the ret instructions in a function match the
; delcared return type of the function they live in.
;
implementation

uint "testfunc"()
begin
	ret int* null
end
