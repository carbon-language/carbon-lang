; RUN: llvm-as < %s | opt -dse | llvm-dis | grep store

double %foo(sbyte* %X) {
        %X_addr = alloca sbyte*
        store sbyte* %X, sbyte** %X_addr  ;; not a dead store.
        %tmp.0 = va_arg sbyte** %X_addr, double
        ret double %tmp.0
}

