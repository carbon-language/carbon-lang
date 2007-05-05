; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 -mcpu=i386 -mattr=+sse
; PR1371

%str = external global [18 x sbyte]

void %test() {
bb.i:
	%tmp.i660 = load <4 x float>* null
	call void (int, ...)* %printf( int 0, sbyte* getelementptr ([18 x sbyte]* %str, int 0, uint 0), double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00 )
	%tmp152.i = load <4 x uint>* null
	%tmp156.i = cast <4 x uint> %tmp152.i to <4 x int>
	%tmp175.i = cast <4 x float> %tmp.i660 to <4 x int>
	%tmp176.i = xor <4 x int> %tmp156.i, < int -1, int -1, int -1, int -1 >
	%tmp177.i = and <4 x int> %tmp176.i, %tmp175.i
	%tmp190.i = or <4 x int> %tmp177.i, zeroinitializer
	%tmp191.i = cast <4 x int> %tmp190.i to <4 x float>
	store <4 x float> %tmp191.i, <4 x float>* null
	ret void
}

declare void %printf(int, ...)
