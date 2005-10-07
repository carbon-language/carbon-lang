; RUN: llvm-as < %s | llc
double %test(bool %X) {
        %Y = cast bool %X to double
        ret double %Y
}
