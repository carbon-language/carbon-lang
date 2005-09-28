; RUN: llvm-as < %s | llc -march=ppc32 | grep eqv | wc -l  | grep 3 &&
; RUN: llvm-as < %s | llc -march=ppc32 | grep andc | wc -l | grep 2 &&
; RUN: llvm-as < %s | llc -march=ppc32 | grep orc | wc -l  | grep 2 &&
; RUN: llvm-as < %s | llc -march=ppc32 | grep nor | wc -l  | grep 2 &&
; RUN: llvm-as < %s | llc -march=ppc32 | grep nand | wc -l  | grep 1

int %EQV1(int %X, int %Y) {
	%A = xor int %X, %Y
	%B = xor int %A, -1
	ret int %B
}

int %EQV2(int %X, int %Y) {
	%A = xor int %X, -1
	%B = xor int %A, %Y
	ret int %B
}

int %EQV3(int %X, int %Y) {
	%A = xor int %X, -1
	%B = xor int %Y, %A
	ret int %B
}

int %ANDC1(int %X, int %Y) {
	%A = xor int %Y, -1
	%B = and int %X, %A
	ret int %B
}

int %ANDC2(int %X, int %Y) {
	%A = xor int %X, -1
	%B = and int %A, %Y
	ret int %B
}

int %ORC1(int %X, int %Y) {
	%A = xor int %Y, -1
	%B = or  int %X, %A
	ret int %B
}

int %ORC2(int %X, int %Y) {
	%A = xor int %X, -1
	%B = or  int %A, %Y
	ret int %B
}

int %NOR1(int %X) {
        %Y = xor int %X, -1
        ret int %Y
}

int %NOR2(int %X, int %Y) {
        %Z = or int %X, %Y
        %R = xor int %Z, -1
        ret int %R
}

int %NAND1(int %X, int %Y) {
	%Z = and int %X, %Y
	%W = xor int %Z, -1
	ret int %W
}
