; RUN: llvm-upgrade < %s | llvm-as | llvm-dis | grep 1.0

double %test() {
        ret double 1.0   ;; This should not require hex notation
}

