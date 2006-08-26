; RUN: llvm-as < %s | llc -march=x86 -relocation-model=static | not grep 'subl.*%esp'

%A = external global ushort*
%B = external global uint
%C = external global uint

void %test() {
	%tmp = load ushort** %A
	%tmp1 = getelementptr ushort* %tmp, int 1
	%tmp = load ushort* %tmp1
	%tmp3 = cast ushort %tmp to uint
	%tmp = load uint* %B
	%tmp4 = and uint %tmp, 16
	%tmp5 = load uint* %C
	%tmp6 = cast uint %tmp4 to ubyte
	%tmp7 = shl uint %tmp5, ubyte %tmp6
	%tmp9 = xor ubyte %tmp6, 16
	%tmp11 = shr uint %tmp3, ubyte %tmp9
	%tmp12 = or uint %tmp11, %tmp7
	store uint %tmp12, uint* %C
	ret void
}
