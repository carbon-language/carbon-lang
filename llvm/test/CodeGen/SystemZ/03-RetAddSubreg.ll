; RUN: llvm-as < %s | llc -march=systemz | grep ar    | count 3
; RUN: llvm-as < %s | llc -march=systemz | grep lgfr  | count 3
; RUN: llvm-as < %s | llc -march=systemz | grep llgfr | count 2

define i32 @foo(i32 %a, i32 %b) {
entry:
    %c = add i32 %a, %b
    ret i32 %c
}

define i32 @foo1(i32 %a, i32 %b) zeroext {
entry:
    %c = add i32 %a, %b
    ret i32 %c
}

define i32 @foo2(i32 %a, i32 %b) signext {
entry:
    %c = add i32 %a, %b
    ret i32 %c
}

