; RUN: llvm-as < %s | llc -march=ppc32 | grep eqv | wc -l  | grep 3 &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | grep andc | wc -l | grep 3 &&
; RUN: llvm-as < %s | llc -march=ppc32 | grep orc | wc -l  | grep 2 &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | grep nor | wc -l  | grep 3 &&
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

void %VNOR(<4 x float>* %P, <4 x float>* %Q) {
        %tmp = load <4 x float>* %P
        %tmp = cast <4 x float> %tmp to <4 x int>
        %tmp2 = load <4 x float>* %Q
        %tmp2 = cast <4 x float> %tmp2 to <4 x int>
        %tmp3 = or <4 x int> %tmp, %tmp2
        %tmp4 = xor <4 x int> %tmp3, < int -1, int -1, int -1, int -1 >
        %tmp4 = cast <4 x int> %tmp4 to <4 x float>
        store <4 x float> %tmp4, <4 x float>* %P
        ret void
}

void %VANDC(<4 x float>* %P, <4 x float>* %Q) {
        %tmp = load <4 x float>* %P
        %tmp = cast <4 x float> %tmp to <4 x int>
        %tmp2 = load <4 x float>* %Q
        %tmp2 = cast <4 x float> %tmp2 to <4 x int>
        %tmp4 = xor <4 x int> %tmp2, < int -1, int -1, int -1, int -1 >
        %tmp3 = and <4 x int> %tmp, %tmp4
        %tmp4 = cast <4 x int> %tmp3 to <4 x float>
        store <4 x float> %tmp4, <4 x float>* %P
        ret void
}

