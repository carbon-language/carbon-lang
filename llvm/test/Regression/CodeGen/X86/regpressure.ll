;; Both functions in this testcase should codegen to the same function, and
;; neither of them should require spilling anything to the stack.

; RUN: llvm-as < %s | llc -march=x86 -disable-pattern-isel=0 -stats  2>&1 | not grep 'Number of register spills'

;; This can be compiled to use three registers if the loads are not
;; folded into the multiplies, 2 registers otherwise.
int %regpressure1(int* %P) {
	%A = load int* %P
	%Bp = getelementptr int* %P, int 1
	%B = load int* %Bp
	%s1 = mul int %A, %B
	%Cp = getelementptr int* %P, int 2
	%C = load int* %Cp
	%s2 = mul int %s1, %C
	%Dp = getelementptr int* %P, int 3
	%D = load int* %Dp
	%s3 = mul int %s2, %D
	%Ep = getelementptr int* %P, int 4
	%E = load int* %Ep
	%s4 = mul int %s3, %E
	%Fp = getelementptr int* %P, int 5
	%F = load int* %Fp
	%s5 = mul int %s4, %F
	%Gp = getelementptr int* %P, int 6
	%G = load int* %Gp
	%s6 = mul int %s5, %G
	%Hp = getelementptr int* %P, int 7
	%H = load int* %Hp
	%s7 = mul int %s6, %H
	%Ip = getelementptr int* %P, int 8
	%I = load int* %Ip
	%s8 = mul int %s7, %I
	%Jp = getelementptr int* %P, int 9
	%J = load int* %Jp
	%s9 = mul int %s8, %J
	ret int %s9
}

;; This testcase should produce identical code to the test above.
int %regpressure2(int* %P) {
	%A = load int* %P
	%Bp = getelementptr int* %P, int 1
	%B = load int* %Bp
	%Cp = getelementptr int* %P, int 2
	%C = load int* %Cp
	%Dp = getelementptr int* %P, int 3
	%D = load int* %Dp
	%Ep = getelementptr int* %P, int 4
	%E = load int* %Ep
	%Fp = getelementptr int* %P, int 5
	%F = load int* %Fp
	%Gp = getelementptr int* %P, int 6
	%G = load int* %Gp
	%Hp = getelementptr int* %P, int 7
	%H = load int* %Hp
	%Ip = getelementptr int* %P, int 8
	%I = load int* %Ip
	%Jp = getelementptr int* %P, int 9
	%J = load int* %Jp
	%s1 = mul int %A, %B
	%s2 = mul int %s1, %C
	%s3 = mul int %s2, %D
	%s4 = mul int %s3, %E
	%s5 = mul int %s4, %F
	%s6 = mul int %s5, %G
	%s7 = mul int %s6, %H
	%s8 = mul int %s7, %I
	%s9 = mul int %s8, %J
	ret int %s9
}

