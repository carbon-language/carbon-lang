; RUN: llvm-as < %s | llc -march=sparcv9

void %main() {
	%tmp.0.i2.i = malloc [24 x sbyte]
	ret void
}
