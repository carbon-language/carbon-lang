; RUN: as < %s | opt -raise | dis | grep -v uint | not grep 4294967295

%length_code = uninitialized global [256 x ubyte]

ubyte* %test(uint %length) {
	%d = add uint 4294967295, %length
	%e = cast uint %d to int
	%g = cast int %e to ulong
	%j = cast [256 x ubyte]* %length_code to ulong
	%l = add ulong %j, %g
	%m = cast ulong %l to ubyte*
	ret ubyte* %m
}
