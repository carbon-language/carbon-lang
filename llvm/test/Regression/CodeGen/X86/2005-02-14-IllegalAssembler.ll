; RUN: llvm-as < %s | llc | not grep 18446744073709551612

%A = external global int

%Y = global int* getelementptr (int* %A, int -1)
