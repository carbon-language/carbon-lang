; RUN: llvm-as < %s | llc -march=systemz | grep nill | count 3
; RUN: llvm-as < %s | llc -march=systemz | grep nilh | count 3

define i32 @foo1(i32 %a, i32 %b) {
entry:
    %c = and i32 %a, 1
    ret i32 %c
}

define i32 @foo2(i32 %a, i32 %b) {
entry:
    %c = and i32 %a, 131072
    ret i32 %c
}

define i32 @foo3(i32 %a, i32 %b) zeroext {
entry:
    %c = and i32 %a, 1
    ret i32 %c
}

define i32 @foo4(i32 %a, i32 %b) signext {
entry:
    %c = and i32 %a, 131072
    ret i32 %c
}

define i32 @foo5(i32 %a, i32 %b) zeroext {
entry:
    %c = and i32 %a, 1
    ret i32 %c
}

define i32 @foo6(i32 %a, i32 %b) signext {
entry:
    %c = and i32 %a, 131072
    ret i32 %c
}

