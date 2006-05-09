; Coalescing from R32 to a subset R32_. Once another register coalescer bug is
; fixed, the movb should go away as well.

; RUN: llvm-as < %s | llc -march=x86 -relocation-model=static | grep 'movl' | wc -l

%B = external global uint
%C = external global ushort*

void %test(uint %A) {
	%A = cast uint %A to ubyte
	%tmp2 = load uint* %B
	%tmp3 = and ubyte %A, 16
	%tmp4 = shl uint %tmp2, ubyte %tmp3
	store uint %tmp4, uint* %B
	%tmp6 = shr uint %A, ubyte 3
	%tmp = load ushort** %C
	%tmp8 = cast ushort* %tmp to uint
	%tmp9 = add uint %tmp8, %tmp6
	%tmp9 = cast uint %tmp9 to ushort*
	store ushort* %tmp9, ushort** %C
	ret void
}
