; RUN: if as < %s | opt -raise | dis | grep 4294967295
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

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
