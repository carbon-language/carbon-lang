
%MyVar     = uninitialized global int
%MyIntList = global { \2 *, int } { { \2, int }* null, int 17 }
             uninitialized global int      ; int*:0

%AConst    = constant int 123

declare int "foo"(int %blah)      ;; Declared in testlink2.ll

declare void "print"(int %Value)

implementation

void "main"()
begin
	%v1 = load int* %MyVar
	call void %print(int %v1)    ;; Should start out 4

	%v2 = load { \2 *, int }* %MyIntList, ubyte 1
	call void %print(int %v2)    ;; Should start out 17

	call int %foo(int 5)         ;; Modify global variablesx

	%v3 = load int* %MyVar
	call void %print(int %v3)    ;; Should now be 5

	%v4 = load { \2 *, int }* %MyIntList, ubyte 1
	call void %print(int %v4)    ;; Should start out 12

	ret void
end

