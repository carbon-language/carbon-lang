int %main() {
	%A = add sbyte 0, 12
	%B = sub sbyte %A, 1
	%C = mul sbyte %B, %B
	%D = div sbyte %C, %C
	%E = rem sbyte %D, %D
	%F = div ubyte 5, 6
	%G = rem ubyte 6, 5

	%A = add short 0, 12
	%B = sub short %A, 1
	%C = mul short %B, %B
	%D = div short %C, %C
	%E = rem short %D, %D
	%F = div ushort 5, 6
	%G = rem uint 6, 5

	%A = add int 0, 12
	%B = sub int %A, 1
	%C = mul int %B, %B
	%D = div int %C, %C
	%E = rem int %D, %D
	%F = div uint 5, 6
	%G = rem uint 6, 5

	%A = add long 0, 12
	%B = sub long %A, 1
	%C = mul long %B, %B
	%D = div long %C, %C
	%E = rem long %D, %D
	%F = div ulong 5, 6
	%G = rem ulong 6, 5

	ret int 0
}
