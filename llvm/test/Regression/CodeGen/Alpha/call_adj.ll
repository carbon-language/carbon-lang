;All this should do is not crash
;RUN: llvm-as < %s | llc -march=alpha

target endian = little
target pointersize = 64
target triple = "alphaev67-unknown-linux-gnu"

implementation   ; Functions:

void %_ZNSt13basic_filebufIcSt11char_traitsIcEE22_M_convert_to_externalEPcl(uint %f) {
entry:
	%tmp49 = alloca sbyte, uint %f		; <sbyte*> [#uses=1]
	%tmp = call uint null( sbyte* null, sbyte* null, sbyte* null, sbyte* null, sbyte* null, sbyte* null, sbyte* null)
	ret void
}
