; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86

void %_ZN13QFSFileEngine4readEPcx() {
	%tmp201 = load int* null
	%tmp201 = cast int %tmp201 to long
	%tmp202 = load long* null
	%tmp203 = add long %tmp201, %tmp202
	store long %tmp203, long* null
	ret void
}
