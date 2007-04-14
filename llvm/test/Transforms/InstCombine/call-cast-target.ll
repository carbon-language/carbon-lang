; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   grep call | not grep bitcast

target endian = little
target pointersize = 32
target triple = "i686-pc-linux-gnu"

implementation   ; Functions:

int %main() {
entry:
	%tmp = call int cast (sbyte* (int*)* %ctime to int (int*)*)( int* null )
	ret int %tmp
}

declare sbyte* %ctime(int*)
