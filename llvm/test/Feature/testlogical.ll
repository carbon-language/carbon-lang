implementation

int "simpleAdd"(int %i0, int %j0)
begin
	%t1 = xor int %i0, %j0
	%t2 = or int %i0, %j0
	%t3 = and int %t1, %t2
	ret int %t3
end

