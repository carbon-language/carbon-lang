
global %MyVar     = int *
global %MyIntList = { \2 *, int } *
global int *     ; int*:0

implementation

int "foo"(int %blah)
begin
	store int 5, int *%MyVar
	store int 12, { \2 *, int } * %MyIntList, ubyte 1
	ret int %blah
end

