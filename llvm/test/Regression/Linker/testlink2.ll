
%MyVar     = global int 4
%MyIntList = uninitialized global { \2 *, int }

%AConst    = constant int 123

implementation

int "foo"(int %blah)
begin
	store int %blah, int *%MyVar
	store int 12, { \2 *, int } * %MyIntList, ubyte 1
	ret int %blah
end

