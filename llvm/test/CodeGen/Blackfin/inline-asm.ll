; RUN: llc < %s -march=bfin | FileCheck %s

; Standard "r"
; CHECK: r0 = r0 + r1;
define i32 @add_r(i32 %A, i32 %B) {
	%R = call i32 asm "$0 = $1 + $2;", "=r,r,r"( i32 %A, i32 %B ) nounwind
	ret i32 %R
}

; Target "d"
; CHECK: r0 = r0 - r1;
define i32 @add_d(i32 %A, i32 %B) {
	%R = call i32 asm "$0 = $1 - $2;", "=d,d,d"( i32 %A, i32 %B ) nounwind
	ret i32 %R
}

; Target "a" for P-regs
; CHECK: p0 = (p0 + p1) << 1;
define i32 @add_a(i32 %A, i32 %B) {
	%R = call i32 asm "$0 = ($1 + $2) << 1;", "=a,a,a"( i32 %A, i32 %B ) nounwind
	ret i32 %R
}

; Target "z" for P0, P1, P2. This is not a real regclass
; CHECK: p0 = (p0 + p1) << 2;
define i32 @add_Z(i32 %A, i32 %B) {
	%R = call i32 asm "$0 = ($1 + $2) << 2;", "=z,z,z"( i32 %A, i32 %B ) nounwind
	ret i32 %R
}

; Target "C" for CC. This is a single register
; CHECK: cc = p0 < p1;
; CHECK: r0 = cc;
define i32 @add_C(i32 %A, i32 %B) {
	%R = call i32 asm "$0 = $1 < $2;", "=C,z,z"( i32 %A, i32 %B ) nounwind
	ret i32 %R
}

