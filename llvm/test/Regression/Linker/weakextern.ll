; RUN: llvm-as < %s > %t.bc
; RUN: llvm-as < `dirname %s`/testlink1.ll > %t2.bc
; RUN: llvm-link %t.bc %t.bc %t2.bc -o %t1.bc -f
; RUN: llvm-dis < %t1.bc |grep "kallsyms_names = extern_weak" &&
; RUN: llvm-dis < %t1.bc |grep "MyVar = external global int" &&
; RUN: llvm-dis < %t1.bc |grep "Inte = global int"

%kallsyms_names = extern_weak global [0 x ubyte]
%MyVar     = extern_weak global int
%Inte = extern_weak global int

implementation
