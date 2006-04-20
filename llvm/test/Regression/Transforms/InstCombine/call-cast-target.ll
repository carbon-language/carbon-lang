; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep call | not grep cast

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
