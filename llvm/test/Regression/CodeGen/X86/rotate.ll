; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=intel | grep ro[rl] | wc -l | grep 12

uint %rotl32(uint %A, ubyte %Amt) {
	%B = shl uint %A, ubyte %Amt
	%Amt2 = sub ubyte 32, %Amt
	%C = shr uint %A, ubyte %Amt2
	%D = or uint %B, %C
	ret uint %D
}

uint %rotr32(uint %A, ubyte %Amt) {
	%B = shr uint %A, ubyte %Amt
	%Amt2 = sub ubyte 32, %Amt
	%C = shl uint %A, ubyte %Amt2
	%D = or uint %B, %C
	ret uint %D
}

uint %rotli32(uint %A) {
	%B = shl uint %A, ubyte 5
	%C = shr uint %A, ubyte 27
	%D = or uint %B, %C
	ret uint %D
}

uint %rotri32(uint %A) {
	%B = shr uint %A, ubyte 5
	%C = shl uint %A, ubyte 27
	%D = or uint %B, %C
	ret uint %D
}

ushort %rotl16(ushort %A, ubyte %Amt) {
	%B = shl ushort %A, ubyte %Amt
	%Amt2 = sub ubyte 16, %Amt
	%C = shr ushort %A, ubyte %Amt2
	%D = or ushort %B, %C
	ret ushort %D
}

ushort %rotr16(ushort %A, ubyte %Amt) {
	%B = shr ushort %A, ubyte %Amt
	%Amt2 = sub ubyte 16, %Amt
	%C = shl ushort %A, ubyte %Amt2
	%D = or ushort %B, %C
	ret ushort %D
}

ushort %rotli16(ushort %A) {
	%B = shl ushort %A, ubyte 5
	%C = shr ushort %A, ubyte 11
	%D = or ushort %B, %C
	ret ushort %D
}

ushort %rotri16(ushort %A) {
	%B = shr ushort %A, ubyte 5
	%C = shl ushort %A, ubyte 11
	%D = or ushort %B, %C
	ret ushort %D
}

ubyte %rotl8(ubyte %A, ubyte %Amt) {
	%B = shl ubyte %A, ubyte %Amt
	%Amt2 = sub ubyte 8, %Amt
	%C = shr ubyte %A, ubyte %Amt2
	%D = or ubyte %B, %C
	ret ubyte %D
}

ubyte %rotr8(ubyte %A, ubyte %Amt) {
	%B = shr ubyte %A, ubyte %Amt
	%Amt2 = sub ubyte 8, %Amt
	%C = shl ubyte %A, ubyte %Amt2
	%D = or ubyte %B, %C
	ret ubyte %D
}

ubyte %rotli8(ubyte %A) {
	%B = shl ubyte %A, ubyte 5
	%C = shr ubyte %A, ubyte 3
	%D = or ubyte %B, %C
	ret ubyte %D
}

ubyte %rotri8(ubyte %A) {
	%B = shr ubyte %A, ubyte 5
	%C = shl ubyte %A, ubyte 3
	%D = or ubyte %B, %C
	ret ubyte %D
}
