; RUN: llvm-as < %s | llc -march=x86 | not grep 18446744073709551612

%A = external global int

%Y = global int* getelementptr (int* %A, int -1)
