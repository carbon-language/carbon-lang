; RUN: llvm-as < %s | llc -march=c


%version = global [3 x sbyte] c"1\00\00"

