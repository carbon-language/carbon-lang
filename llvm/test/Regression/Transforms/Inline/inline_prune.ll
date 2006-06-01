; RUN: llvm-as < %s | opt -inline -disable-output &&
; RUN: llvm-as < %s | opt -inline | llvm-dis | not grep 'callee[12](' &&
; RUN: llvm-as < %s | opt -inline | llvm-dis | not grep mul

implementation

internal int %callee1(int %A, int %B) {
	%cond = seteq int %A, 123
	br bool %cond, label %T, label %F
T:
	%C = mul int %B, %B
	ret int %C
F:
	ret int 0
}

internal int %callee2(int %A, int %B) {
	switch int %A, label %T [
          int 10, label %F
          int 1234, label %G
        ]
	%cond = seteq int %A, 123
	br bool %cond, label %T, label %F
T:
	%C = mul int %B, %B
	ret int %C
F:
	ret int 0
G:
	%D = mul int %B, %B
	%E = mul int %D, %B
	ret int %E
}

int %test(int %A) {
	%X = call int %callee1(int 10, int %A)
	%Y = call int %callee2(int 10, int %A)
	%Z = add int %X, %Y
	ret int %Z
}
