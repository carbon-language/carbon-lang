
int %main() {
	%A = and sbyte 4, 8
	%B = or sbyte %A, 7
	%C = xor sbyte %B, %A

	%A = and short 4, 8
	%B = or short %A, 7
	%C = xor short %B, %A

	%A = and int 4, 8
	%B = or int %A, 7
	%C = xor int %B, %A

	%A = and long 4, 8
	%B = or long %A, 7
	%C = xor long %B, %A

	ret int 0
}
