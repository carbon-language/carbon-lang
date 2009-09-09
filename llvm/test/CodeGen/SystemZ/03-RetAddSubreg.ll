; RUN: llc < %s -march=systemz | grep ar    | count 3
; RUN: llc < %s -march=systemz | grep lgfr  | count 2
; RUN: llc < %s -march=systemz | grep llgfr | count 1

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

