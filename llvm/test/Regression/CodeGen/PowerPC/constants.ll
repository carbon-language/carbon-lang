; All of these routines should be perform optimal load of constants.
; RUN: llvm-as < constants.ll | llc -march=ppc32 | grep lis   | wc -l | grep 5 &&
; RUN: llvm-as < constants.ll | llc -march=ppc32 | grep ori   | wc -l | grep 3 &&
; RUN: llvm-as < constants.ll | llc -march=ppc32 | grep 'li ' | wc -l | grep 4

implementation   ; Functions:

int %_Z2f1v() {
entry:
        ret int 1
}

int %_Z2f2v() {
entry:
        ret int -1
}

int %_Z2f3v() {
entry:
        ret int 0
}

int %_Z2f4v() {
entry:
        ret int 32767
}

int %_Z2f5v() {
entry:
        ret int 65535
}

int %_Z2f6v() {
entry:
        ret int 65536
}

int %_Z2f7v() {
entry:
        ret int 131071
}

int %_Z2f8v() {
entry:
        ret int 2147483647
}

int %_Z2f9v() {
entry:
        ret int -2147483648
}
