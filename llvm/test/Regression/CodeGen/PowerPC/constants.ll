; All of these routines should be perform optimal load of constants.
; RUN: llvm-as < %s | llc -march=ppc32 | grep lis   | wc -l | grep 5 &&
; RUN: llvm-as < %s | llc -march=ppc32 | grep ori   | wc -l | grep 3 &&
; RUN: llvm-as < %s | llc -march=ppc32 | grep 'li ' | wc -l | grep 4

implementation   ; Functions:

int %f1() {
entry:
        ret int 1
}

int %f2() {
entry:
        ret int -1
}

int %f3() {
entry:
        ret int 0
}

int %f4() {
entry:
        ret int 32767
}

int %f5() {
entry:
        ret int 65535
}

int %f6() {
entry:
        ret int 65536
}

int %f7() {
entry:
        ret int 131071
}

int %f8() {
entry:
        ret int 2147483647
}

int %f9() {
entry:
        ret int -2147483648
}
