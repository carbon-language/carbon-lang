implementation
int "testExpressions"(int %N, int* %A)
begin
	%N1 = add int %N, 12
	%N2 = mul int %N, 7
	%N3 = add int %N1, %N2    ;; Should equal 8*N+12
	%N4 = shl int %N3, ubyte 3      ;; Should equal 64*N + 96
	%N5 = mul int %N4, 0      ;; 0
	br label %1

	%C  = cast int 264 to ubyte  ;; 8
	%C1 = add ubyte 252, %C      ;; 4
	%C2 = cast ubyte %C1 to ulong ;; 4
	%C3 = add ulong 12345678901, %C2 ;; 12345678905
	%C4 = cast ulong %C3 to sbyte * ;; 12345678905
	br label %2

	%A1 = cast int 4 to int *
	%A2 = add int *%A, %A1    ;; %A+4
	%A3 = cast int 8 to int *
	%A4 = add int *%A, %A3    ;; %A+8
	%X  = sub int *%A4, %A2   ;; Should equal 4
	br label %3

	%Z1 = cast int 400 to int *
	%Z2 = cast sbyte 2 to int *
	%Z3 = add int* %Z1, %Z2
	%Z4 = cast int* %Z3 to ubyte

	ret int %N4
end

