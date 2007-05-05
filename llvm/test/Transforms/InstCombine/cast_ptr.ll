; Tests to make sure elimination of casts is working correctly
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | notcast

target pointersize = 32

implementation

sbyte* %test1(sbyte* %t) {
	%tmpc = cast sbyte* %t to uint
	%tmpa = add uint %tmpc, 32
	%tv = cast uint %tmpa to sbyte*
	ret sbyte* %tv
}

bool %test2(sbyte* %a, sbyte* %b) {
%tmpa = cast sbyte* %a to uint
%tmpb = cast sbyte* %b to uint
%r = seteq uint %tmpa, %tmpb
ret bool %r
}
