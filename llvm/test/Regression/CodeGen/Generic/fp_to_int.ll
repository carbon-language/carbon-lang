; RUN: llvm-as < %s | llc

sbyte %test1(double %X) {
	%tmp.1 = cast double %X to sbyte
	ret sbyte %tmp.1
}
short %test2(double %X) {
	%tmp.1 = cast double %X to short
	ret short %tmp.1
}
int %test3(double %X) {
	%tmp.1 = cast double %X to int
	ret int %tmp.1
}
long %test4(double %X) {
	%tmp.1 = cast double %X to long
	ret long %tmp.1
}
ubyte %test1u(double %X) {
	%tmp.1 = cast double %X to ubyte
	ret ubyte %tmp.1
}
ushort %test2u(double %X) {
	%tmp.1 = cast double %X to ushort
	ret ushort %tmp.1
}
uint %test3u(double %X) {
	%tmp.1 = cast double %X to uint
	ret uint %tmp.1
}
ulong %test4u(double %X) {
	%tmp.1 = cast double %X to ulong
	ret ulong %tmp.1
}

sbyte %test1f(float %X) {
	%tmp.1 = cast float %X to sbyte
	ret sbyte %tmp.1
}
short %test2f(float %X) {
	%tmp.1 = cast float %X to short
	ret short %tmp.1
}
int %test3f(float %X) {
	%tmp.1 = cast float %X to int
	ret int %tmp.1
}
long %test4f(float %X) {
	%tmp.1 = cast float %X to long
	ret long %tmp.1
}
ubyte %test1uf(float %X) {
	%tmp.1 = cast float %X to ubyte
	ret ubyte %tmp.1
}
ushort %test2uf(float %X) {
	%tmp.1 = cast float %X to ushort
	ret ushort %tmp.1
}
uint %test3uf(float %X) {
	%tmp.1 = cast float %X to uint
	ret uint %tmp.1
}
ulong %test4uf(float %X) {
	%tmp.1 = cast float %X to ulong
	ret ulong %tmp.1
}
