; RUN: llvm-upgrade < %s | llvm-as | llc -march=alpha | grep cmpbge | wc -l | grep 2

bool %test1(ulong %A, ulong %B) {
	%C = and ulong %A, 255
	%D = and ulong %B, 255
	%E = setge ulong %C, %D
	ret bool %E
}

bool %test2(ulong %a, ulong %B) {
	%A = shl ulong %a, ubyte 1
	%C = and ulong %A, 254
	%D = and ulong %B, 255
	%E = setge ulong %C, %D
	ret bool %E
}
