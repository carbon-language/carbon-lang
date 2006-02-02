; RUN: llvm-as < %s | llc -march=x86 &&
; RUN: llvm-as < %s | llc -march=x86 | not grep add
bool %X(int %X) {
        %Y = add int %X, 14
        %Z = setne int %Y, 12345
        ret bool %Z
}

