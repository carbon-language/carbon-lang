
void %test() {
	%A = add sbyte 0, 12
	%B = sub sbyte %A, %A
	%C = mul sbyte %B, %B
	%D = div sbyte %C, %C
	%E = rem sbyte %D, %D
	%F = div ubyte 5, 6
	%G = rem ubyte 6, 5

	%A = add short 0, 12
	%B = sub short %A, %A
	%C = mul short %B, %B
	%D = div short %C, %C
	%E = rem short %D, %D
	%F = div ushort 5, 6
	%G = rem uint 6, 5

	%A = add int 0, 12
	%B = sub int %A, %A
	%C = mul int %B, %B
	%D = div int %C, %C
	%E = rem int %D, %D
	%F = div uint 5, 6
	%G = rem uint 6, 5

	ret void
}
