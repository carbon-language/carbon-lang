%somestr = constant [sbyte] c"hello world"
%somestr = constant [11x sbyte] c"hello world"
%array   = constant [[2 x int]]  [ [2 x int] [ int 12, int 52 ] ]
           constant { int, int } { int 4, int 3 }

implementation
 
[[2 x int]]* "test function"(int %i0, int %j0)
begin
	ret [[2x int]]* %array
end

[sbyte]* "other func"(int, double)
begin
	ret [sbyte]* %somestr
end

[sbyte]* "yet another func"(int, double)
begin
	ret [sbyte]* null            ; Test null
end


[sbyte]* "again"(float)
begin
	%cast = cast [11x sbyte]* %somestr to [sbyte]*
	ret [sbyte]* %cast
end

