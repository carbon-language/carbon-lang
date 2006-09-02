; RUN: llvm-as < %s | opt -argpromotion | llvm-dis | grep x.val
; ModuleID = 'recursive2.bc'

implementation   ; Functions:

internal int %foo(int* %x) {
entry:
        %tmp = load int* %x
        %tmp.foo = call int %foo(int *%x)
        ret int %tmp.foo
}

int %bar(int* %x) {
entry:
        %tmp3 = call int %foo( int* %x)                ; <int>[#uses=1]
        ret int %tmp3
}