%somestr = [sbyte] c"hello world"
%somestr = [11x sbyte] c"hello world"

implementation
 
[[2 x int]] "test function"(int %i0, int %j0)
        %array = [[2 x int]] [
                   [2 x int] [ int 12, int 52 ]
                 ]
begin
	ret [[2x int]] %array
end


[sbyte] "other func"(int, double)
begin
	ret [sbyte] %somestr
end

[sbyte] "again"(float)
begin
	%cast = cast [11x sbyte] %somestr to [sbyte]
	ret [sbyte] %cast
end
