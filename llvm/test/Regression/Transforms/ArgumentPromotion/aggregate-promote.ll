; RUN: llvm-as < %s | opt -argpromotion -instcombine | not grep load

%QuadTy = type {int, int, int, int}

%G = constant %QuadTy {int 0, int 0, int 17, int 25 }

implementation

internal int %test(%QuadTy* %P) {
	%A = getelementptr %QuadTy* %P, long 0, ubyte 3
	%B = getelementptr %QuadTy* %P, long 0, ubyte 2
	%a = load int* %A
	%b = load int* %B
	%V = add int %a, %b
	ret int %V
}

int %caller() {
	%V = call int %test(%QuadTy* %G)
	ret int %V
}
