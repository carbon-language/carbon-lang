
%MyVar     = uninitialized global int
%MyIntList = uninitialized global { \2 *, int }
             external global int      ; int*:0

%AConst    = constant int 123

%AString   = constant [4 x ubyte] c"test"

%ZeroInit  = global { [100 x int ], [40 x float ] } { [100 x int] zeroinitializer,
                                                      [40  x float] zeroinitializer }

implementation

int "foo"(int %blah)
begin
	store int 5, int *%MyVar
	%idx = getelementptr { \2 *, int } * %MyIntList, long 0, ubyte 1
  	store int 12, int* %idx
  	ret int %blah
end

