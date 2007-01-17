; RUN: llvm-upgrade < %s | llvm-as | llc -march=c

; Test const pointer refs & forward references

%t3 = global int * %t1           ;; Forward reference
%t1 = global int 4

