
%MyVar     = uninitialized global int
%MyIntList = uninitialized global { \2 *, int }
             uninitialized global int      ; int*:0

%AConst    = constant int 123

%AString   = constant [4 x ubyte] c"test"

implementation

int "foo"(int %blah)
begin
	store int 5, int *%MyVar
	%idx = getelementptr { \2 *, int } * %MyIntList, uint 0, ubyte 1
  	store int 12, int* %idx
  	ret int %blah
end

