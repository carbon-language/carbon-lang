; RUN: llvm-upgrade < %s | llvm-as > %t.bc
; RUN: llvm-upgrade < %p/testlink1.ll | llvm-as > %t2.bc
; RUN: llvm-link %t.bc %t.bc %t2.bc -o %t1.bc -f
; RUN: llvm-dis < %t1.bc | grep {kallsyms_names = extern_weak}
; RUN: llvm-dis < %t1.bc | grep {MyVar = external global i32}
; RUN: llvm-dis < %t1.bc | grep {Inte = global i32}

%kallsyms_names = extern_weak global [0 x ubyte]
%MyVar     = extern_weak global int
%Inte = extern_weak global int

implementation
