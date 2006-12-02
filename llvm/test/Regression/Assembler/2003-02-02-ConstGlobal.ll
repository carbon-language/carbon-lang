; RUN: llvm-upgrade < %s | llvm-as -o /dev/null -f

%X = external global int
%X = constant int 7
