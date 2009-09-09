; RUN: llc < %s -march=systemz | grep ngr | count 3
; RUN: llc < %s -march=systemz | grep nihf | count 1

define i32 @foo(i32 %a, i32 %b) {
entry:
    %c = and i32 %a, %b
    ret i32 %c
}

define i32 @foo1(i32 %a, i32 %b) zeroext {
entry:
    %c = and i32 %a, %b
    ret i32 %c
}

define i32 @foo2(i32 %a, i32 %b) signext {
entry:
    %c = and i32 %a, %b
    ret i32 %c
}

