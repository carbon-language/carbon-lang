
%MyVar     = global int 4
%MyIntList = uninitialized global { \2 *, int }

%AConst    = constant int 123

constant int 412

implementation

int "foo"(int %blah)
begin
	store int %blah, int *%MyVar
	store int 12, { \2 *, int } * %MyIntList, ubyte 1

	;%ack = load int * %0   ;; Load from the unnamed constant
	;%fzo = add int %ack, %blah
	;ret int %fzo
	ret int %blah
end

declare void "unimp"(float, double)
