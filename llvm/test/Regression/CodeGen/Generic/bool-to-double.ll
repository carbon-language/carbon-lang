; RUN: llvm-upgrade < %s | llvm-as | llc
double %test(bool %X) {
        %Y = cast bool %X to double
        ret double %Y
}
